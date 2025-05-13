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
from .rpc_handler import RPCHandler


def _extract_job_times(response: dict) -> tuple[float, float, float]:
    creation_datetime = response.get('creation_datetime')
    start_datetime = response.get('start_time')
    duration = response.get('duration')
    if creation_datetime is not None:
        creation_datetime = float(creation_datetime)
    if start_datetime is not None:
        start_datetime = float(start_datetime)
    if duration is not None:
        duration = float(duration)
    return creation_datetime, duration, start_datetime


class RemoteJob(Job):
    """
    Handle a computation task remotely (i.e. through a Cloud provider) by sending a request body of the form:

    .. code-block::

        {
            'platform_name': '...',
            'pcvl_version': 'M.m.p',
            'payload': {
                'command': '...',
                'circuit': <optional serialized circuit>,
                'input_state': <optional serialized input state>,
                ...
                other parameters
            }
        }

    :param request_data: prepared data for the job. It is extended by an execute_async() call before being sent.
                         It is expected to be prepared by a RemoteProcessor.
    :param rpc_handler: a valid RPC handler to connect to a Cloud provider
    :param job_name: the job name (visible cloud-side)
    :param delta_parameters: parameters to add/remove dynamically
    :param job_context: Data on the job execution context (conversion required on results, etc.)
    :param command_param_names: List of parameter names for the command call (in order to resolve \*args in
                                execute_async() call). This parameter is optional (default = empty list). However,
                                without it, only \*\*kwargs are available in the execute_async() call
    :param refresh_progress_delay: wait time when running in sync mode between each refresh (in seconds)
    """

    STATUS_REFRESH_DELAY = 1  # minimum job status refresh period (in s)
    _MAX_ERROR = 5

    def __init__(self, request_data: dict, rpc_handler: RPCHandler, job_name: str,
                 delta_parameters: dict = None, job_context: dict = None,
                 command_param_names: list = None, refresh_progress_delay: int = 3):
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
    def was_sent(self) -> bool:
        """
        :return: True if the job was sent to a Cloud provider, False otherwise
        """
        return self._id is not None  # id is created when created on a Cloud

    @property
    def id(self) -> str | None:
        """
        Job unique identifier

        :return: a UUID, or None if the job was never sent to a Cloud provider
        """
        return self._id

    @staticmethod
    def from_id(job_id: str, rpc_handler: RPCHandler) -> RemoteJob:
        """
        Recreate an existing RemoteJob from its unique identifier, and a RPCHandler.

        :param job_id: existing unique identifier
        :param rpc_handler: a RPCHandler with valid credentials to connect to the Cloud provider where the job was sent
        """
        rj = RemoteJob(None, rpc_handler, "resumed")  # There is no access to the job name, let's call it "resumed"
        rj._id = job_id
        rj.status()
        return rj

    @staticmethod
    def _from_dict(my_dict: dict, rpc_handler):
        if my_dict['status'] == 'SUCCESS':
            body = None
            name = ""
        else:
            body = my_dict['body']
            name = my_dict['body']['job_name']
        rj = RemoteJob(body, rpc_handler, name)
        rj._id = my_dict['id']
        if my_dict['status'] is not None:
            rj._job_status.status = RunningStatus[my_dict['status']]
        return rj

    def _to_dict(self):
        job_info = dict()
        job_info['id'] = self.id
        job_info['status'] = str(self._job_status) if self.was_sent else None

        # save metadata to recreate remote jobs
        job_info['metadata'] = {'headers': self._rpc_handler.headers,
                                'platform': self._rpc_handler.name,
                                'url': self._rpc_handler.url,
                                'proxies': self._rpc_handler.proxies}

        if not self._job_status.success:
            # Save the job payload for later use in the cloud unless the status is 'success'.
            job_info['body'] = self._create_payload_data()

        return job_info

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
        if not self.was_sent or self._job_status.completed:  # static status - retrieving from the cloud unnecessary.
            return self._job_status

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
                self._job_status._stop_message = response['status_message']

            creation_datetime, duration, start_datetime = _extract_job_times(response)
            self._job_status.update_times(creation_datetime, start_datetime, duration)

        return self._job_status

    def execute_sync(self, *args, **kwargs) -> dict:
        """
        Execute the task synchronously. This call is blocking and immediately returns a results dictionary.

        .. note:: This method has exactly the same effect as __call__.

        .. warning:: A remote job natural way of running is asynchronous. This call will actually run the task
                     asynchronously with a waiting loop to reproduce the behaviour of the synchronous call.

        :param args: arguments to pass to the task function
        :param kwargs: keyword arguments to pass to the task function
        :return: the results
        """
        job = self.execute_async(*args, **kwargs)
        while not job.is_complete:
            time.sleep(self._refresh_progress_delay)
        return self.get_results()

    def _create_payload_data(self, *args, **kwargs) -> dict:
        # creates the payload for the job and returns the prepared job data
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

    def execute_async(self, *args, **kwargs) -> RemoteJob:
        assert self._job_status.waiting, "job has already been executed"
        try:
            self._id = self._rpc_handler.create_job(serialize(self._create_payload_data(*args, **kwargs)))
            get_logger().info(f"Send payload to the Cloud (got job id: {self._id})", channel.general)
            self._job_status.status = RunningStatus.WAITING

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
        Job must have failed, meaning job status is either CANCELED or ERROR.

        :raises RuntimeError: Job have not failed, therefore it cannot be rerun.
        :return: The new remote job.
        """
        if not self.status.failed:
            raise RuntimeError(
                f"Cannot rerun current job because job status is: {self.status} (should be either CANCELED or ERROR)")

        job_dict = self._to_dict()
        job_dict['id'] = self._rpc_handler.rerun_job(self._id)
        job_dict['status'] = RunningStatus.WAITING.name
        return RemoteJob._from_dict(job_dict, self._rpc_handler)

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
        if not self.was_sent:
            return f"RemoteJob '{self.name}', id:{self._id}, status:not sent"
        else:
            return f"RemoteJob '{self.name}', id:{self._id}, status:{self._job_status}"
