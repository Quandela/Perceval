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
from threading import Thread
from typing import Callable

from perceval.utils import ProgressCallback
from perceval.components import Experiment

from .abstract_computer import AbstractComputer
from .computation import Computation


class ThreadedExecution:
    """AsyncGetter - Private class"""

    def __init__(self, method: Callable, args: tuple=(), kwargs: dict = None):
        self._thread = Thread(target=self._encapsulate(method), args=args, kwargs=kwargs)
        self._results = None
        self._current_progress = 0
        self._progress_message = ""
        self._canceled = False
        self._thread.start()

    def _encapsulate(self, method: Callable):
        def custom_method(*args, **kwargs):
            try:
                self._results = method(*args, **kwargs, progress_cb = self._progress_cb)
            except TypeError as e:
                if "progress_cb" in str(e):
                    self._results = method(*args, **kwargs)
                else:
                    raise e

            self._current_progress = 1
            self._progress_message = "Finished!"

        return custom_method

    def get_results(self):
        self._thread.join()
        return self._results

    def cancel(self):
        self._canceled = True

    def get_progress(self):
        return self._current_progress

    def _progress_cb(self, progress: float, message: str) -> bool:
        self._progress_message = message
        self._current_progress = progress
        return self._canceled

    def is_complete(self) -> bool:
        return not self._thread.is_alive()


class LocalComputer(AbstractComputer, ABC):

    def __init__(self):
        super().__init__()
        self._commands = ["probs", "samples", "sample_count"]

    def _execute_command(self, computation: Computation, progress_cb: ProgressCallback = None) -> dict:
        return getattr(self, computation.command.name)(computation.experiment, progress_cb, **computation.parameters)

    def _execute_command_async(self, computation: Computation) -> ThreadedExecution:
        return ThreadedExecution(self._execute_single, args=(computation,))

    def _load_async_result(self, async_getter: ThreadedExecution) -> dict:
        return async_getter.get_results()

    def is_complete(self, async_getter: ThreadedExecution) -> bool:
        return async_getter.is_complete()

    @abstractmethod
    def probs(self, experiment: Experiment, progress_cb: ProgressCallback = None, **kwargs) -> dict:
        pass

    @abstractmethod
    def samples(self, experiment: Experiment, progress_cb: ProgressCallback = None, **kwargs) -> dict:
        pass

    @abstractmethod
    def sample_count(self, experiment: Experiment, progress_cb: ProgressCallback = None, **kwargs) -> dict:
        pass

    @property
    def is_remote(self) -> bool:
        return False
