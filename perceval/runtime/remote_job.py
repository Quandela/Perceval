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
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import time
from typing import Any, Callable

from .job import Job
from .job_status import JobStatus, RunningStatus
from .rpc_handler import RPCHandler


class RemoteJob(Job):
    def __init__(self, fn: Callable, rpc_handler: RPCHandler, deserializer: Callable):
        super().__init__(fn)
        self._rpc_handler = rpc_handler
        self._deserializer = deserializer
        self._job_status = JobStatus()

    @property
    def status(self) -> JobStatus:
        response = self._rpc_handler.get_job_status(self._id)
        self._job_status.status = RunningStatus.from_server_response(response['status'])
        if self._job_status.status == RunningStatus.RUNNING:
            self._job_status.update_progress(float(response['progress']))
        # TODO get _init_time_start and _completed_time from server response
        return self._job_status

    def execute_sync(self, *args, **kwargs) -> Any:
        assert self._job_status.waiting, "job as already been executed"
        job = self.execute_async(*args, **kwargs)
        while not job.is_completed():
            time.sleep(3)
        return self.get_results()

    def execute_async(self, *args, **kwargs):
        assert self._job_status.waiting, "job as already been executed"
        try:
            self._id = self._fn(*args, **kwargs)
        except Exception as e:
            self._job_status.stop_run(RunningStatus.ERROR, str(e))

        return self

    def get_results(self) -> Any:
        response = self._rpc_handler.get_job_results(self._id)
        results = response['results']

        if self._deserializer is not None:
            return self._deserializer(results)
        else:
            return results
