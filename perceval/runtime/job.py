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
from typing import Dict, Callable

from .job_status import JobStatus, RunningStatus


class Job(ABC):
    def __init__(self, result_mapping_function: Callable = None, delta_parameters=None, command_param_names=None):
        self._results = None
        self._result_mapping_function = result_mapping_function
        self._delta_parameters = delta_parameters or {"command": {}, "mapping": {}}
        self._param_names = command_param_names or []
        self._name = "Job"

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, new_name: str):
        if not isinstance(new_name, str):
            raise TypeError("A job name must be a string")
        self._name = new_name if len(new_name) > 0 else "unnamed"

    def _handle_params(self, args, kwargs):
        """Handle args and kwargs parameters from __call__, execute_sync or execute_async
        Split parameters between the command and the post-process function (mapping),
        See: self._delta_parameters and self._param_names
        """
        args = list(args)
        if len(args) > len(self._param_names):
            self._delta_parameters['mapping']['max_samples'] = args.pop()
        for idx, unnamed_arg in enumerate(args):
            param_name = self._param_names[idx]
            if param_name in kwargs:  # Parameter exists twice (in args and in kwargs)
                raise RuntimeError(f'Parameter named {param_name} was passed twice (in *args and **kwargs)')
            self._delta_parameters['command'][param_name] = unnamed_arg
        for k, v in self._delta_parameters['command'].items():
            if v is None and k in kwargs:
                self._delta_parameters['command'][k] = kwargs[k]
                del kwargs[k]
        for k, v in self._delta_parameters['mapping'].items():
            if v is None and k in kwargs:
                self._delta_parameters['mapping'][k] = kwargs[k]
                del kwargs[k]
        if kwargs:
            raise RuntimeError(f"Unused parameters in user call ({list(kwargs.keys())})")

    def __call__(self, *args, **kwargs) -> Dict:
        return self.execute_sync(*args, **kwargs)

    @abstractmethod
    def get_results(self) -> Dict:
        pass

    @property
    @abstractmethod
    def status(self) -> JobStatus:
        pass

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
    def execute_sync(self, *args, **kwargs) -> Dict:
        pass

    @abstractmethod
    def execute_async(self, *args, **kwargs):
        pass

    @abstractmethod
    def cancel(self):
        pass
