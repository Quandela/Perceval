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

from time import time, sleep
from typing import Optional
from enum import Enum


class RunningStatus(Enum):
    WAITING = 0
    RUNNING = 1
    SUCCESS = 2
    ERROR = 3
    CANCELED = 4
    SUSPENDED = 5

    @staticmethod
    def from_server_response(res):
        if res == 'completed':
            return RunningStatus.SUCCESS
        else:
            return RunningStatus[res.upper()]


class JobStatus:
    def __init__(self):
        self._status: RunningStatus = RunningStatus.WAITING
        self._init_time_start = time()
        self._running_time_start = None
        self._completed_time = None
        self._running_progress: float = 0
        self._running_phase: Optional[str] = None
        self._stop_message = None
        self._waiting_progress: Optional[int] = None
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

    def stop_run(self, cause: RunningStatus = RunningStatus.SUCCESS, mesg: Optional[str] = None):
        self._status = cause
        self._completed_time = time()
        if cause == RunningStatus.SUCCESS:
            self._running_progress = 1
        self._stop_message = mesg

    def update_progress(self, progress: float, phase: Optional[str] = None):
        if self._status == RunningStatus.WAITING:
            self.start_run()
        self._running_progress = progress
        self._running_phase = phase
        now = time()
        if self._last_progress_time is not None and self._last_progress_time - now > 5:
            sleep(0.001)
        self._last_progress_time = now

    def update_times(self, start_time: float, duration: int):
        if start_time is None:
            return

        self._init_time_start = start_time
        if duration is not None:
            self._completed_time = start_time + duration

    @property
    def waiting(self):
        return self._status == RunningStatus.WAITING

    @property
    def running(self):
        return self._status == RunningStatus.RUNNING

    @property
    def completed(self):
        return self._status in [RunningStatus.SUCCESS, RunningStatus.ERROR, RunningStatus.CANCELED]

    @property
    def success(self):
        return self._status == RunningStatus.SUCCESS

    @property
    def failed(self):
        return self._status in [RunningStatus.CANCELED, RunningStatus.ERROR]

    @property
    def stop_message(self):
        return self._stop_message

    @property
    def progress(self):
        return self._running_progress

    @property
    def running_time(self):
        assert self.completed
        return self._completed_time-self._running_time_start
