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

from enum import Enum
from time import time, sleep

from perceval.utils.logging import get_logger, channel


class RunningStatus(Enum):
    WAITING = 0
    RUNNING = 1
    SUCCESS = 2
    ERROR = 3
    CANCELED = 4
    SUSPENDED = 5
    CANCEL_REQUESTED = 6
    UNKNOWN = 7

    @staticmethod
    def from_server_response(res: str) -> RunningStatus:
        """Converts a job status name from the server to an enum value.

        .. note:: the server name for SUCCESS is "completed".

        :param res: the job status name
        :return: the corresponding enum value or UNKNOWN if the status name is unknown.
        """
        if res == 'completed':
            return RunningStatus.SUCCESS
        else:
            try:
                return RunningStatus[res.upper()]
            except KeyError:
                get_logger().warn(f"Unknown job running status: {res}", channel.user)
                return RunningStatus.UNKNOWN

    @staticmethod
    def to_server_response(status: RunningStatus) -> str:
        """
        Converts a job status enum value to an acceptable name for the server.

        .. note:: SUCCESS is converted to "completed".

        :param status: the job status enum value
        :return: the status name
        """
        if status == RunningStatus.SUCCESS:
            return 'completed'
        else:
            return status.name.lower()

RunningStatus.WAITING.__doc__ = ("The job is recorded on the Cloud but waits for a computing platform to be available "
                                 "in order to start.")
RunningStatus.RUNNING.__doc__ = "The job is being run on a given computing platform."
RunningStatus.SUCCESS.__doc__ = "The job has completed successfully. The full results are to be expected."
RunningStatus.ERROR.__doc__ = "The job has failed and partial results might be available."
RunningStatus.CANCELED.__doc__ = ("The job was canceled either by the user or the system. "
                                  "Partial results might be available.")
RunningStatus.SUSPENDED.__doc__ = "The job was halted by the remote system and may be resumed later on."
RunningStatus.CANCEL_REQUESTED.__doc__ = ("Transitional status leading to CANCELED. "
                                          "The Cloud sent a cancel order to the platform running the job.")
RunningStatus.UNKNOWN.__doc__ = "An unknown status code was encountered."


class JobStatus:
    """
    Stores metadata related to a job execution
    """
    def __init__(self):
        self._status: RunningStatus = RunningStatus.WAITING
        self._init_time_start = time()
        self._running_time_start: float | None = None
        self._duration: float | None = None
        self._completed_time: float | None = None
        self._running_progress: float = 0
        self._running_phase: str | None = None
        self._stop_message: str | None = None
        self._waiting_progress: int | None = None
        self._last_progress_time: float = 0

    def __call__(self) -> str:
        """
        Return the name of the running status

        :return: name of the status
        """
        return self._status.name

    @property
    def status(self) -> RunningStatus:
        """
        :return: the job running status
        """
        return self._status

    @status.setter
    def status(self, status: RunningStatus):
        self._status = status

    def start_run(self):
        """
        Informs that the job is starting.
        Sets the job start time as the current time and the running status to "RUNNING"
        """
        self._running_time_start = time()
        self._status = RunningStatus.RUNNING

    def stop_run(self, cause: RunningStatus = RunningStatus.SUCCESS, mesg: str | None = None):
        """
        Informs that the job has just stopped.
        Sets the job stop time as the current time.

        :param cause: running status causing the end of the job
        :param mesg: optional additional message related to the end of the job
        """
        self._status = cause
        self._completed_time = time()
        self._duration = self._completed_time - self._init_time_start
        if cause == RunningStatus.SUCCESS:
            self._running_progress = 1
        self._stop_message = mesg

    def update_progress(self, progress: float, phase: str | None = None):
        """
        Updates the job progress.

        :param progress: the current progress (between 0 and 1, 1 meaning 100%)
        :param phase: message related to the current progress
        """
        if self._status == RunningStatus.WAITING:
            self.start_run()
        self._running_progress = progress
        self._running_phase = phase
        now = time()
        if self._last_progress_time and self._last_progress_time - now > 5:
            sleep(0.001)
        self._last_progress_time = now

    def update_times(self, creation_datetime: float, start_time: float, duration: float):
        """
        Set the important times from external information

        :param creation_datetime: the timestamp the job was created
        :param start_time: the timestamp the job was started
        :param duration: the duration of the job (in seconds)
        """
        self._init_time_start = creation_datetime
        self._running_time_start = start_time
        self._duration = duration
        if self.completed:
            self._completed_time = self._running_time_start + self._duration

    @property
    def creation_timestamp(self) -> float:
        """
        :return: the timestamp the job was created
        """
        return self._init_time_start

    @property
    def start_timestamp(self) -> float:
        """
        :return: the timestamp the job was started
        """
        return self._running_time_start

    @property
    def duration(self) -> float:
        """
        :return: the duration of the job (in seconds)
        """
        return self._duration

    @property
    def waiting(self) -> bool:
        """
        :return: whether the job is in "WAITING" status
        """
        return self._status == RunningStatus.WAITING

    @property
    def running(self) -> bool:
        """
        :return: whether the job is running (corresponding statuses are "RUNNING" and "CANCEL_REQUESTED")
        """
        return self._status in [RunningStatus.RUNNING, RunningStatus.CANCEL_REQUESTED]

    @property
    def completed(self) -> bool:
        """
        :return: whether the job has completed, i.e. not waiting or running anymore (corresponding statuses are
                 "SUCCESS", "ERROR" and "CANCELED")
        """
        return self._status in [RunningStatus.SUCCESS, RunningStatus.ERROR, RunningStatus.CANCELED]

    @property
    def canceled(self) -> bool:
        """
        :return: whether the job is in "CANCELED" status
        """
        return self._status in [RunningStatus.CANCELED]

    @property
    def success(self) -> bool:
        """
        :return: whether the job is in "SUCCESS" status
        """
        return self._status in [RunningStatus.SUCCESS]

    @property
    def failed(self) -> bool:
        """
        :return: whether the job has failed to complete (corresponding statuses are "CANCELED" and "ERROR")
        """
        return self._status in [RunningStatus.CANCELED, RunningStatus.ERROR]

    @property
    def maybe_completed(self) -> bool:
        """
        :return: whether the job has or might have completed (corresponding statuses are "SUCCESS", "ERROR", "CANCELED"
                 and "UNKNOWN")
        """
        return self._status in [RunningStatus.SUCCESS, RunningStatus.ERROR, RunningStatus.CANCELED, RunningStatus.UNKNOWN]

    @property
    def unknown(self) -> bool:
        """
        :return: whether the job status is unknown
        """
        return self._status in [RunningStatus.UNKNOWN]

    @property
    def stop_message(self) -> str | None:
        """
        :return: the job stop message, if any. In case of a successful job, this will be `None`.
        """
        return self._stop_message

    @property
    def progress(self) -> float:
        """
        :return: the current job progress (between 0 and 1, 1 meaning 100%)
        """
        return self._running_progress

    @property
    def running_time(self) -> float:
        if self._duration:
            return self._duration
        if not self.completed:
            return time() - self._running_time_start
        return self._completed_time - self._running_time_start

    def __str__(self) -> str:
        return self._status.name
