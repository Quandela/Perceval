# MIT License
#
# Copyright (c) 2022 Quandela
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# As a special exception, the copyright holders of exqalibur library give you
# permission to combine exqalibur with code included in the standard release of
# Perceval under the MIT license (or modified versions of such code). You may
# copy and distribute such a combined system following the terms of the MIT
# license for both exqalibur and Perceval. This exception for the usage of
# exqalibur is limited to the python bindings used by Perceval.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from __future__ import annotations

import json
import time
from requests.exceptions import HTTPError, ConnectionError

from .job import Job
from .job_status import JobStatus, RunningStatus
from perceval.serialization import deserialize, serialize
from perceval.utils.logging import get_logger, channel


def _extract_job_times(response: dict):
    creation_datetime = response.get('creation_datetime')
    start_datetime = response.get('start_time')
    duration = response.get('duration')
    if creation_datetime is not None:
        creation_datetime = float(creation_datetime)
    if start_datetime is not None:
        start_datetime = float(start_datetime)
    if duration is not None:
        duration = int(duration)
    return creation_datetime, duration, start_datetime


class RemoteJob(Job):
    STATUS_REFRESH_DELAY = 1  # minimum job status refresh period (in s)
    _MAX_ERROR = 5

    def __init__(self, request_data, rpc_handler, job_name, delta_parameters=None, job_context=None,
                 command_param_names=None, refresh_progress_delay: int = 3):
        r"""
        :param request_data: prepared data for the job. It is extended by an async_execute() call before being sent.
            It is expected to be prepared by a RemoteProcessor. It must have the following structure:
            {
              'platform_name': '...',
              'pcvl_version': 'M.m.p',
              'payload': {
                'command': '...',
                'circuit': <optional serialized circuit>,
                'input_state': <optional serialized input state>
                ...
                other parameters
              }
            }
        :param rpc_handler: a valid RPC handler to connect to the cloud
        :param job_name: the job name (visible cloud-side)
        :param delta_parameters: parameters to add/remove dynamically
        :param job_context: Data on the job execution context (conversion required on results, etc.)
        :param command_param_names: List of parameter names for the command call (in order to resolve *args in
            async_execute() call). This parameter is optional (default = empty list). However, without it, only **kwargs
            are available in the async_execute() call
        :param refresh_progress_delay: wait time when running in sync mode between each refresh
        """
        super().__init__(delta_parameters=delta_parameters, command_param_names=command_param_names)
        self._rpc_handler = rpc_handler
        self._job_status = JobStatus()
        self._job_context = job_context
        self._refresh_progress_delay = refresh_progress_delay  # When syncing an async job (in s)
        self._previous_status_refresh = 0.
        self._status_refresh_error = 0
        self._id = None
        self.name = job_name
        self._request_data = request_data
        self._results = None

    @property
    def id(self):
        return self._id

    @staticmethod
    def from_id(job_id: str, rpc_handler):
        j = RemoteJob(None, rpc_handler, "resumed")  # There is no access to the job name, let's call it "resumed"
        j._id = job_id
        j.status()
        return j

    def _handle_status_error(self, error):
        """
        Handle a potentially non-blocking error
        After _MAX_ERROR errors in a row, the exception is raised
        """
        self._status_refresh_error += 1
        if self._status_refresh_error == self._MAX_ERROR:
            get_logger().error("Reached max number of HTTP errors in a row when updating job {self._id} status.",
                               channel.general)
            raise error
        if isinstance(error, HTTPError):
            error_code = error.response.status_code
            if error_code in [
                408,  # Time-out
                409,  # Conflict in the current state of the resource
                421,  # Misdirected request
                423,  # Resource locked
                429   # Too many requests
            ]:
                get_logger().error(f"Got HTTP error {error_code} when updating job {self._id} status. Ignoring...",
                             channel.general)
            else:  # If the status code is any other error, it is considered unrecoverable
                raise error
        return self._job_status

    @property
    def status(self) -> JobStatus:
        now = time.time()
        if now - self._previous_status_refresh > RemoteJob.STATUS_REFRESH_DELAY:
            self._previous_status_refresh = now
            try:
                response = self._rpc_handler.get_job_status(self._id)
                self._status_refresh_error = 0
            except (HTTPError, ConnectionError) as error:
                self._handle_status_error(error)
                return self._job_status  # Return previous status

            self._job_status.status = RunningStatus.from_server_response(response['status'])
            if self._job_status.running:
                self._job_status.update_progress(float(response['progress']), response['progress_message'])
            elif self._job_status.failed:
                self._job_status._stop_message = response['failure_code']

            creation_datetime, duration, start_datetime = _extract_job_times(response)
            self._job_status.update_times(creation_datetime, start_datetime, duration)

            name = response.get("name")
            if name and name != self.name:
                self.name = name

        return self._job_status

    def execute_sync(self, *args, **kwargs) -> any:
        job = self.execute_async(*args, **kwargs)
        while not job.is_complete:
            time.sleep(self._refresh_progress_delay)
        return self.get_results()

    def _create_payload_data(self, *args, **kwargs):
        self._handle_params(args, kwargs)
        if self._delta_parameters['mapping']:
            if self._job_context is None:
                self._job_context = {}
            self._job_context["mapping_delta_parameters"] = self._delta_parameters["mapping"]

        kwargs = self._delta_parameters['command']
        kwargs['job_context'] = self._job_context
        self._request_data['job_name'] = self._name
        self._request_data['payload'].update(kwargs)
        self._check_max_shots_samples_validity()
        return self._request_data

    def execute_async(self, *args, **kwargs):
        assert self._job_status.waiting, "job has already been executed"
        try:
            self._id = self._rpc_handler.create_job(serialize(self._create_payload_data(*args, **kwargs)))
            get_logger().info(f"Send payload to the Cloud (got job id: {self._id})", channel.general)

        except Exception as e:
            self._job_status.stop_run(RunningStatus.ERROR, str(e))
            raise e

        return self

    def _check_max_shots_samples_validity(self):
        p = self._request_data['payload']
        if "max_samples" in p and "max_shots" in p:
            if p["max_samples"] > p["max_shots"]:
                get_logger().warn(f"Lowered 'max_samples' from user defined value ({p['max_samples']}) to 'max_shots' value ({p['max_shots']}) for consistency.",
                            channel.user)
                p["max_samples"] = p["max_shots"]

    def cancel(self):
        if self.status.status in (RunningStatus.RUNNING, RunningStatus.WAITING, RunningStatus.SUSPENDED):
            get_logger().info(f"Programmatically request job {self._id} cancellation", channel.general)
            self._rpc_handler.cancel_job(self._id)
            self._job_status.stop_run(RunningStatus.CANCEL_REQUESTED, 'Cancellation requested by user')
        else:
            raise RuntimeError('Job is not waiting or running, cannot cancel it')

    def rerun(self) -> RemoteJob:
        """Rerun a job. Same job will be executed again as a new one.
        Job must have failed, meaning job status must be either CANCELED or ERROR.

        :raises RuntimeError: Job have not failed, therefore it cannot be rerun.
        :return: The new remote job.
        """
        if not self.status.failed:
            raise RuntimeError(f"Cannot rerun current job because job status is: {self.status} (should be either CANCELED or ERROR)")
        return RemoteJob.from_id(self._rpc_handler.rerun_job(self._id), self._rpc_handler)


    def _get_results(self) -> None:
        if self._results and self.status.completed:
            return self._results
        response = self._rpc_handler.get_job_results(self._id)
        self._results = deserialize(json.loads(response['results']))
        if "job_context" in self._results and 'result_mapping' in self._results["job_context"]:
            path_parts = self._results["job_context"]["result_mapping"]
            get_logger().info(f"Converting job {self._id} results with {path_parts[1]}", channel.general)
            module = __import__(path_parts[0], fromlist=path_parts[1])
            result_mapping_function = getattr(module, path_parts[1])
            # retrieve delta parameters from the response
            self._delta_parameters = self._results["job_context"].get(
                "mapping_delta_parameters", {})
            if "results_list" in self._results:
                for res in self._results["results_list"]:
                    mapping_args = {key: res["iteration"].get(key, val) for key, val in self._delta_parameters.items()}

                    res["results"] = result_mapping_function(res['results'], **mapping_args)
            else:
                self._results["results"] = result_mapping_function(
                    self._results["results"], **self._delta_parameters)
        return self._results

    def __str__(self):
        if self._id is None:
            # handles unsent jobs of a JobGroup
            return f"RemoteJob with name:{self.name}, id:{self._id}, status:None"
        else:
            return f"RemoteJob with name:{self.name}, id:{self._id}, status:{self._job_status}"
