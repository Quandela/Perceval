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

from abc import ABC, abstractmethod

from .job_status import JobStatus

from perceval.utils.logging import channel, get_logger


class AsyncGetter(ABC):
    """
    Descriptor of an asynchronously launched computation, through its means to get the results.
    Each Computer can implement its own version of this class, depending on its needs.

    Methods can also be added to extend this class for a particular implementation.

    The execution of the computation must begin when this class is initialized.

    Implementations of this class are not meant to be used directly by the user, but only hidden by an upper layer.
    """

    def __init__(self):
        self._results = None
        self._status = JobStatus()

    @property
    def status(self) -> JobStatus:
        """
        The job status metadata structure
        """
        self._update_status()
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

    @abstractmethod
    def cancel(self):
        """
        Request the cancellation of the execution.
        """

    @abstractmethod
    def _get_results(self):
        """Implemented get_results()"""

    def get_results(self) -> dict:
        """
        Retrieve the results of the execution.

        :return: results dictionary. You can expect a "results" or a "results_list" field, performance scores and other
                 data corresponding to the job nature.
        :raises: RuntimeError if the job hasn't finished running, or if the results data are empty or malformed.
        """
        job_status = self.status

        if job_status.canceled:
            get_logger().warn("Job has been canceled, trying to get partial result.", channel.user)

        if job_status.unknown:
            get_logger().warn("Unknown job status, trying to get result anyway.", channel.user)

        try:
            return self._get_results()
        except (KeyError, TypeError):
            if job_status.failed:
                raise RuntimeError(f'The job failed: {job_status.stop_message}')
            else:
                raise RuntimeError('Results are not available')

    @abstractmethod
    def _update_status(self) -> None:
        pass
