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
from enum import Enum
from typing import Any, Callable, Dict, List

from perceval.utils import BasicState, Parameter


class ProcessorType(Enum):
    SIMULATOR = 1
    PHYSICAL = 2


class AProcessor(ABC):
    def __init__(self):
        self._parameters = {}

    @property
    @abstractmethod
    def m(self) -> int:
        pass

    @property
    @abstractmethod
    def type(self) -> ProcessorType:
        pass

    @property
    @abstractmethod
    def is_remote(self) -> bool:
        pass

    def set_parameters(self, params: Dict):
        self._parameters = params

    def set_parameter(self, key: str, value: Any):
        self._parameters[key] = value

    def clear_parameters(self):
        self._parameters = {}

    def mode_post_selection(self, n: int):
        self.set_parameter('mode_post_select', n)

    @abstractmethod
    def with_input(self, input_state: BasicState) -> None:
        pass

    @property
    @abstractmethod
    def available_commands(self) -> List[str]:
        pass
