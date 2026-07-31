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

from abc import ABC
from threading import Thread
from typing import Callable, Any

from perceval.utils import ProgressCallback, parse_signature
from perceval.components import Experiment

from .job_status import RunningStatus
from .abstract_computer import AbstractComputer
from .computation import Computation
from .async_getter import AsyncGetter
from .command import Command


class _ThreadedExecution(AsyncGetter):
    """Async execution for local computer - Private class"""

    def __init__(self, method: Callable, args: tuple=(), kwargs: dict = None):
        super().__init__()
        self._thread = Thread(target=self._encapsulate(method), args=args, kwargs=kwargs)
        self._canceled = False
        self._user_callback: ProgressCallback | None = None  # Do we want to pass a user callback if this is async ?
        self._thread.start()

    def _update_status(self) -> None:
        if self._status.running and not self._thread.is_alive():
            if self._canceled:
                self._status.stop_run(RunningStatus.CANCELED, "Canceled")
            else:
                self._status.stop_run()

    def _encapsulate(self, method: Callable):
        def custom_method(*args, **kwargs):
            self.status.start_run()
            try:
                try:
                    self._results = method(*args, **kwargs, progress_callback = self._progress_callback)
                except TypeError as e:
                    if "progress_callback" in str(e):
                        self._results = method(*args, **kwargs)
                    else:
                        raise e

                if not self._status.canceled:
                    self._status.stop_run()

            except Exception as e:
                msg = f"{type(e).__name__}: {e}"
                self._results = {"results": msg}
                self._status.stop_run(RunningStatus.ERROR, msg)
                raise e  # Keep this ? This makes messages I'm unable to catch in the tests

        return custom_method

    def _get_results(self):
        self._thread.join()
        return self._results

    def cancel(self):
        self._status.stop_run(RunningStatus.CANCELED, "Canceled")
        self._canceled = True

    def get_progress(self):
        return self._status.progress

    def _progress_callback(self, progress: float, message: str) -> bool:
        self._status.update_progress(progress, message)
        if self._canceled:
            return True
        if self._user_callback is not None:
            return self._user_callback(progress, message)
        return self._canceled

    def is_complete(self) -> bool:
        return not self._thread.is_alive()


class LocalComputer(AbstractComputer, ABC):
    """An abstract computer for local computer. Must implement at least "probs", "sample_count", and "samples" methods."""

    def __init__(self):
        super().__init__()
        self._methods: dict[str, Callable[[LocalComputer, Experiment, ...], dict]] = {}

    def _register_method(self,
                         method: Callable[[Any, Experiment, ...], dict],
                         name: str = None,
                         use_emt: bool = True):
        """
        :param method: The callable to use for this method.
            The first 3 arguments (self, experiment, progress_callback) of this method are ignored in the underlying command
        :param name: The name to give to the command. If none is provided, the method name will be used.
        :param use_emt: Whether this method must skip the error mitigation when called
        """
        signature = parse_signature(method)

        if signature[1] is not None and signature[1] != dict:
            raise TypeError(f"Method {method.__name__} must return a dict.")

        sig = signature[0][2:]  # Removes self and Experiment

        for i, (param_name, _, _) in enumerate(sig):
            if param_name == "progress_callback":
                sig.pop(i)
                break

        name = name or method.__name__
        command = Command(name, sig, use_emt)
        self._register_command(command)
        self._methods[name] = method

    def _execute_command(self, computation: Computation, progress_callback: ProgressCallback = None) -> dict:
        try:
            return self._methods[computation.command.name](self,
                                                           computation.experiment,
                                                           progress_callback=progress_callback,
                                                           **computation.parameters)
        except TypeError as e:
            if "progress_callback" in str(e):
                return self._methods[computation.command.name](self, computation.experiment, **computation.parameters)
            else:
                raise e

    def _execute_command_async(self, computation: Computation) -> _ThreadedExecution:
        return _ThreadedExecution(self._execute_single, args=(computation,))

    @property
    def is_remote(self) -> bool:
        return False
