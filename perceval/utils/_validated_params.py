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

from numbers import Number
from abc import abstractmethod, ABC


class AValidatedParam(ABC):

    def __init__(self, default_value = None):
        if default_value is not None:
            self._validate(default_value)
        self.default_value = default_value

    def __set_name__(self, owner, name):
        self.private_name = '_' + name
        self.name = name

    def __get__(self, obj, objtype=None):
        value = getattr(obj, self.private_name)
        if value is None:
            return self.default_value
        return value

    def __set__(self, obj, value):
        if value is not None:
            self._validate(value)
        setattr(obj, self.private_name, value)

    @abstractmethod
    def _validate(self, value):
        pass

    def is_default(self):
        return self.default_value is not None


class ValidatedBool(AValidatedParam):
    def __init__(self, default_value=None):
        super().__init__(default_value)

    def _validate(self, value):
        if not isinstance(value, bool):
            raise TypeError(f"{self.name} expected a boolean value, got {type(value)}")
        return value


class ValidatedFloat(AValidatedParam):
    def __init__(self, min_value=None, max_value=None, default_value=None):
        self._min = min_value
        self._max = max_value
        super().__init__(default_value)

    def _validate(self, value):
        if not isinstance(value, Number):
            raise TypeError(f"{self.name} expected a numerical value, got {type(value)}")
        if (self._min is None or self._min <= value) and (self._max is None or value <= self._max):
            return value
        raise ValueError(f"{self.name} value out of bound: {value} not in [{self._min}, {self._max}]")
