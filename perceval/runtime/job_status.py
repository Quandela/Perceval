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
from time import time

from perceval.utils.logging import get_logger, channel


class RunningStatus(Enum):
    NONE = -1
    SUCCESS = 0
    WAITING = 1
    RUNNING = 2
    SUSPENDED = 3
    CANCEL_REQUESTED = 4
    CANCELED = 5
    ERROR = 6
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

    @staticmethod
    def merge_with_index(left: RunningStatus, right: RunningStatus, index_left: int, index_right: int) -> tuple[RunningStatus, int]:
        """
        :return: The predominant RunningStatus between left and right, as well as its associated index.
            As a special case, SUCCESS + WAITING = RUNNING, and the returned index is the one of the WAITING status
        """
        # Only exception to the natural order
        if left == RunningStatus.SUCCESS and right == RunningStatus.WAITING:
            return RunningStatus.RUNNING, index_right
        if left == RunningStatus.WAITING and right == RunningStatus.SUCCESS:
            return RunningStatus.RUNNING, index_left

        if left.value < right.value:
            return right, index_right
        return left, index_left

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
        :return: whether the job is in "CANCELED" or "CANCEL_REQUESTED" status
        """
        return self._status in [RunningStatus.CANCELED, RunningStatus.CANCEL_REQUESTED]

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

    def copy_from(self, status: JobStatus):
        self._status = status._status
        if status._init_time_start:
            self._init_time_start = status._init_time_start
        if status._running_time_start:
            self._running_time_start = status._running_time_start
        if status._duration:
            self._duration = status._duration
        if status._completed_time:
            self._completed_time = status._completed_time
        if status._running_progress:
            self._running_progress = status._running_progress
        if status._running_phase:
            self._running_phase = status._running_phase
        if status._stop_message:
            self._stop_message = status._stop_message

    @staticmethod
    def merge_status(status: list[JobStatus]) -> JobStatus:
        res = JobStatus()

        if len(status) == 0:
            return res

        running_status = RunningStatus.NONE
        running_index = 0
        for i, stat in enumerate(status):
            running_status, running_index = RunningStatus.merge_with_index(running_status, stat.status, running_index, i)

        res._status = running_status
        res._init_time_start = min(stat._init_time_start for stat in status)

        running_start_times = [stat._running_time_start for stat in status if stat._running_time_start is not None]
        if len(running_start_times) > 0:
            res._running_time_start = min(running_start_times)

        completed_times = [stat._completed_time for stat in status]
        res._completed_time = max(completed_times) if all(t is not None for t in completed_times) else None

        time_for_duration = res._completed_time if res._completed_time is not None else time()

        if any(stat._duration is not None for stat in status):
            if res._running_time_start is not None:
                res._duration = time_for_duration - res._running_time_start
            else:
                res._duration = time_for_duration - res._init_time_start

        res._running_progress = sum(stat._running_progress for stat in status) / len(status)

        current_maximum_status = status[running_index]
        res._running_phase = current_maximum_status._running_phase
        res._stop_message = current_maximum_status._stop_message

        return res
