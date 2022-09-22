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

from abc import ABC, abstractmethod
from typing import Any, Callable

from .job_status import JobStatus


class Job(ABC):
    def __init__(self, fn: Callable):
        # create an id or check existence of current id
        self._fn = fn
        # id will be assigned by remote job - not implemented for local class
        self._id = None
        self._results = None
        pass

    def __call__(self, *args, **kwargs) -> Any:
        return self.execute_sync(*args, **kwargs)

    @property
    def id(self):
        return self._id

    def get_results(self) -> Any:
        return self._results

    @abstractmethod
    def get_status(self) -> JobStatus:
        pass

    def is_completed(self) -> bool:
        status = self.get_status()
        return status.completed

    @abstractmethod
    def execute_sync(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def execute_async(self, *args, **kwargs) -> None:
        pass
