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

from __future__ import annotations  # Python 3.11 : Replace using Self typing

from copy import deepcopy

from .check_cancel import cancel_requested
from .async_getter import AsyncGetter
from .abstract_computer import AbstractComputer
from .computation import Computation
from .computation_iterator import ComputationIterator
from .job_status import JobStatus, RunningStatus
from .error_mitigation import AbstractMitigation
from perceval.utils import NoiseModel, ProgressCallback


class Execution:

    def __init__(self, computation: Computation | ComputationIterator, computer: AbstractComputer):
        self._computation = deepcopy(computation)
        self._computer = computer
        self._name = computation.job_name
        self._job_group_name = computation.job_group_name
        self._results = {}
        self._status: JobStatus = JobStatus()

        # Storage for async run
        self._mitigations: list[AbstractMitigation] = []
        self._noise: NoiseModel | None = None
        self._getters: list[list[AsyncGetter]] = []

        # Not serialized
        self._user_cb: ProgressCallback | None = None

    def set_progress_callback(self, callback: ProgressCallback):
        """
        Set a progress callback function with the following signature:

        `def progress_callback(progress: float, message: str) -> dict | bool | None`

        If the progress callback returns True, a cancellation is requested.
        This custom callback is only used for synchronous execution.
        For asynchronous execution, call self.cancel() to cancel,
        or self.status, self.is_complete... to monitor the progress.

        :param callback: callback function
        """
        self._user_cb = callback

    def _progress_callback(self, progress: float, phase: str):
        self._status.update_progress(progress, phase)
        if self._user_cb is not None:
            res = self._user_cb(progress, phase)
            if cancel_requested(res):
                self._status.stop_run(RunningStatus.CANCEL_REQUESTED, "Canceled")
            return res
        return None

    @property
    def name(self) -> str:
        """
        The job name
        """
        return self._name

    @name.setter
    def name(self, new_name: str):
        if not isinstance(new_name, str):
            raise TypeError("A job name must be a string")
        self._name = new_name

    @property
    def job_group_name(self) -> str:
        """
        The job name
        """
        return self._name

    @job_group_name.setter
    def job_group_name(self, new_name: str):
        if not isinstance(new_name, str):
            raise TypeError("A job name must be a string")
        self._job_group_name = new_name

    def set_job_group_name(self, new_name: str):  # TODO: legacy; remove ?
        self.job_group_name = new_name

    def _transmit_args(self, *args, **kwargs):
        if len(args) > 0 or len(kwargs) > 0:
            self._computation.add_params(*args, **kwargs)  # Will raise an error if we have a ComputationIterator
        if self._name:
            self._computation.job_name = self._name
        if self._job_group_name:
            self._computation.job_group_name = self._job_group_name

    def __call__(self, *args, **kwargs) -> dict:
        """
        Execute the job synchronously
        """
        return self.execute_sync(*args, **kwargs)

    @property
    def status(self) -> JobStatus:
        """
        The job status metadata structure
        """
        if len(self._getters) > 0 and not self._status.completed:
            all_status = [getter.status for getters in self._getters for getter in getters]
            self._status.copy_from(JobStatus.merge_status(all_status))

        return self._status

    @property
    def is_complete(self) -> bool:
        return self.status.completed

    @property
    def is_failed(self) -> bool:
        return self.status.failed

    @property
    def is_success(self) -> bool:
        return self.status.success

    @property
    def is_waiting(self) -> bool:
        return self.status.waiting

    @property
    def is_running(self) -> bool:
        return self.status.running

    @property
    def was_sent(self) -> bool:  # Name is a legacy. Change it to reflect that an Execution can be local ?
        return len(self._getters) > 0 or len(self._results) > 0

    def cancel(self):
        """
        Request the cancellation of the job.
        """
        if not self.was_sent:
            raise RuntimeError("Execution has not been launched")
        if self.is_complete:
            raise RuntimeError("Execution has already been completed")

        for getters in self._getters:
            for getter in getters:
                try:
                    getter.cancel()
                except RuntimeError:  # Happens if the getter has finished
                    pass

    def clone(self) -> Execution:
        """
        :return: A new execution, identical to this one, except that it is not associated to any run or results,
            and can thus be used with other parameters, or to rerun an execution.
        """
        execution = Execution(deepcopy(self._computation), self._computer)
        execution._name = self._name
        execution._job_group_name = self._job_group_name
        return execution

    def rerun(self) -> Execution:
        """
        :return: A new execution, identical to this one, and run it asynchronously
        """
        if not self.status.failed:
            raise RuntimeError(
                f"Cannot rerun current job because job status is: {self.status} (should be either CANCELED or ERROR)")

        return self.clone().execute_async()

    def execute_sync(self, *args, allow_partial_results: bool = False, **kwargs) -> dict:
        if self._results:
            return self._results  # Problem here if we try to reuse a job with different args and kwargs

        self._status.start_run()
        with self._computer.acquire():
            try:
                self._transmit_args(*args, **kwargs)
                self._computer.execute(self._computation, self._results, progress_callback=self._progress_callback)
            except Exception as e:
                if not allow_partial_results:
                    self._results = {}
                    self._status.stop_run(RunningStatus.ERROR, f"{type(e).__name__}: {e}")
                    raise e
        if self._status.canceled:
            self._status.stop_run(RunningStatus.CANCELED, "Canceled")
        else:
            self._status.stop_run()
        return self._results

    def execute_async(self, *args, **kwargs) -> Execution:
        """
        Execute the task asynchronously. This call is non-blocking allowing for concurrency. Results cannot be expected
        to be ready as soon as this call ends. The results have to be retrieved only when the job status says it's
        completed.

        :param args: arguments to pass to the task function
        :param kwargs: keyword arguments to pass to the task function
        :return: self
        """
        if self.was_sent:
            raise RuntimeError("Execution has already been launched")

        self._status.start_run()
        self._transmit_args(*args, **kwargs)
        self._mitigations, self._noise, self._getters = self._computer.execute_async(self._computation)
        return self

    def get_results(self, allow_partial_results: bool = False) -> dict:
        """
        Retrieve the results of the job.

        :param allow_partial_results: If True, results will be returned even if there is an error somewhere.
                 Else, the error will be raised
        :return: results dictionary. You can expect a "results" or a "results_list" field, performance scores and other
                 data corresponding to the job nature.
        :raises: RuntimeError if the job hasn't been launched, or if there is an error and allow_partial_results is False.
        """
        if not self.was_sent:
            raise RuntimeError("Execution has not been launched")

        if self._results:
            return self._results

        if not allow_partial_results and self.is_failed:
            raise RuntimeError(f"Execution failed: {self._status.stop_message}")

        try:
            self._computer.get_results(self._computation, self._mitigations, self._noise, self._getters, self._results)
        except Exception as e:
            if not allow_partial_results:
                self._results = {}  # Return None as in legacy ?
                raise e
        return self._results

    def __str__(self):
        if not self.was_sent:
            return f"Execution '{self.name}', status:not sent"
        else:
            return f"Execution '{self.name}', status:{self._status}"
