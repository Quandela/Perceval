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
        if status == RunningStatus.SUCCESS:
            return 'completed'
        else:
            return status.name.lower()

class JobStatus:
    def __init__(self):
        self._status: RunningStatus = RunningStatus.WAITING
        self._init_time_start = time()
        self._running_time_start = None
        self._duration = None
        self._completed_time = None
        self._running_progress: float = 0
        self._running_phase: str | None = None
        self._stop_message = None
        self._waiting_progress: int | None = None
        self._last_progress_time: float = 0

    def __call__(self):
        return self._status.name

    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, status: RunningStatus):
        self._status = status

    def start_run(self):
        self._running_time_start = time()
        self._status = RunningStatus.RUNNING

    def stop_run(self, cause: RunningStatus = RunningStatus.SUCCESS, mesg: str | None = None):
        self._status = cause
        self._completed_time = time()
        self._duration = self._completed_time - self._init_time_start
        if cause == RunningStatus.SUCCESS:
            self._running_progress = 1
        self._stop_message = mesg

    def update_progress(self, progress: float, phase: str | None = None):
        if self._status == RunningStatus.WAITING:
            self.start_run()
        self._running_progress = progress
        self._running_phase = phase
        now = time()
        if self._last_progress_time and self._last_progress_time - now > 5:
            sleep(0.001)
        self._last_progress_time = now

    def update_times(self, creation_datetime: float, start_time: float, duration: int):
        if creation_datetime:
            self._init_time_start = creation_datetime
        if start_time:
            self._running_time_start = start_time
        if duration:
            self._duration = duration
            if self.completed:
                self._completed_time = self._running_time_start + self._duration

    @property
    def creation_timestamp(self):
        return self._init_time_start

    @property
    def start_timestamp(self):
        return self._running_time_start

    @property
    def duration(self):
        return self._duration

    @property
    def waiting(self):
        return self._status == RunningStatus.WAITING

    @property
    def running(self):
        return self._status in [RunningStatus.RUNNING, RunningStatus.CANCEL_REQUESTED]

    @property
    def completed(self):
        return self._status in [RunningStatus.SUCCESS, RunningStatus.ERROR, RunningStatus.CANCELED]

    @property
    def canceled(self):
        return self._status in [RunningStatus.CANCELED]

    @property
    def success(self):
        return self._status in [RunningStatus.SUCCESS]

    @property
    def failed(self):
        return self._status in [RunningStatus.CANCELED, RunningStatus.ERROR]

    @property
    def maybe_completed(self):
        return self._status in [RunningStatus.SUCCESS, RunningStatus.ERROR, RunningStatus.CANCELED, RunningStatus.UNKNOWN]

    @property
    def unknown(self):
        return self._status in [RunningStatus.UNKNOWN]

    @property
    def stop_message(self):
        return self._stop_message

    @property
    def progress(self):
        return self._running_progress

    @property
    def running_time(self):
        if self._duration:
            return self._duration

        assert self.completed
        return self._completed_time - self._running_time_start

    def __str__(self) -> str:
        return self._status.name
