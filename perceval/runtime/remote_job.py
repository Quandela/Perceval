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
import json
import time
from typing import Any
from requests.exceptions import HTTPError, ConnectionError

from .job import Job
from .job_status import JobStatus, RunningStatus
from perceval.serialization import deserialize, serialize


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
        super().__init__(delta_parameters=delta_parameters)
        self._rpc_handler = rpc_handler
        self._job_status = JobStatus()
        self._job_context = job_context
        self._refresh_progress_delay = refresh_progress_delay  # When syncing an async job (in s)
        self._previous_status_refresh = 0.
        self._status_refresh_error = 0
        self._id = None
        self.name = job_name
        self._request_data = request_data
        self._param_names = command_param_names or []

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
            raise error
        if isinstance(error, HTTPError):
            error_code = error.response.status_code
            if error_code not in [
                408,  # Time-out
                409,  # Conflict in the current state of the resource
                421,  # Misdirected request
                423,  # Resource locked
                429   # Too many requests
            ]:  # If the status code is any other error, it is considered unrecoverable
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
                return self._handle_status_error(error)

            self._job_status.status = RunningStatus.from_server_response(response['status'])
            if self._job_status.running:
                self._job_status.update_progress(float(response['progress']), response['progress_message'])
            elif self._job_status.failed:
                self._job_status._stop_message = response['failure_code']

            start_time = None
            duration = None
            try:
                start_time = float(response['start_time'])
                duration = int(response['duration'])
            except:
                pass

            self._job_status.update_times(start_time, duration)

        return self._job_status

    def _handle_unnamed_params(self, args, kwargs):
        if len(args) > len(self._param_names):
            raise RuntimeError(f'Too many unnamed parameter: {len(args)}, expected {len(self._param_names)}')
        for idx, unnamed_arg in enumerate(args):
            param_name = self._param_names[idx]
            if param_name in kwargs:  # Parameter exists twice (in args and in kwargs)
                raise RuntimeError(f'Parameter named {param_name} was passed twice (in *args and **kwargs)')
            kwargs[param_name] = unnamed_arg
        return kwargs

    def execute_sync(self, *args, **kwargs) -> Any:
        job = self.execute_async(*args, **kwargs)
        while not job.is_complete:
            time.sleep(self._refresh_progress_delay)
        return self.get_results()

    def execute_async(self, *args, **kwargs):
        assert self._job_status.waiting, "job has already been executed"
        try:
            args, kwargs = self._adapt_parameters(args, kwargs)
            if self._delta_parameters:
                if self._job_context is None:
                    self._job_context = {}
                self._job_context["mapping_delta_parameters"] = self._delta_parameters
            kwargs = self._handle_unnamed_params(args, kwargs)
            kwargs['job_context'] = self._job_context
            self._request_data['job_name'] = self._name
            self._request_data['payload'].update(kwargs)
            self._id = self._rpc_handler.create_job(serialize(self._request_data))

        except Exception as e:
            self._job_status.stop_run(RunningStatus.ERROR, str(e))
            raise e

        return self

    def cancel(self):
        if self.status.status in (RunningStatus.RUNNING, RunningStatus.WAITING, RunningStatus.SUSPENDED):
            self._rpc_handler.cancel_job(self._id)
            self._job_status.stop_run(RunningStatus.CANCEL_REQUESTED, 'Cancelation requested by user')
        else:
            raise RuntimeError('Job is not waiting or running, cannot cancel it')

    def get_results(self) -> Any:
        job_status = self.status
        if not job_status.completed:
            raise RuntimeError('The job is still running, results are not available yet.')
        if job_status.status == RunningStatus.ERROR:
            raise RuntimeError(f'The job failed: {job_status.stop_message}')
        response = self._rpc_handler.get_job_results(self._id)
        results = deserialize(json.loads(response['results']))
        if "job_context" in results and 'result_mapping' in results["job_context"]:
            path_parts = results["job_context"]["result_mapping"]
            module = __import__(path_parts[0], fromlist=path_parts[1])
            result_mapping_function = getattr(module, path_parts[1])
            # retrieve delta parameters from the response
            self._delta_parameters = results["job_context"].get("mapping_delta_parameters", {})
            results["results"] = result_mapping_function(results["results"], **self._delta_parameters)
        return results
