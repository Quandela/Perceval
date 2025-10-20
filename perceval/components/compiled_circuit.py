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
import copy
from packaging.version import Version
from perceval.components.abstract_component import AParametrizedComponent
from perceval.utils.parameter import Parameter


class ChipParameters:
    def __init__(self, chipName: str, params: list[Parameter], version: Version | None = None):
        self.chipName = chipName
        self._params = params
        self.version = version

    def get_values(self) -> list[float] | None:
        result = None
        if all(p.defined for p in self._params):
            result = []
            for p in self._params:
                result.append(p.evalf())
        return result

    def set_values(self, values: list[float]) -> None:
        if len(values) != len(self._params):
            raise "invalid parameters list"
        for p, v in zip(self._params, values):
            p.set_value(v, force = True)

    def need_compilation(self, other: ChipParameters) -> bool:
        if self.chipName != other.chipName:
            raise ValueError("Chip mismatch")
        if not (self.version and other.version):
            return False

        # Consider checking closeness of values
        return self.version != other.version


class CompiledCircuit(AParametrizedComponent):
    def __init__(self, m: int, chipParameters: ChipParameters):
        super().__init__(m, chipParameters.chipName)
        self._chipParameters = chipParameters
        for p in chipParameters._params:
            self._set_parameter(p.name, copy.deepcopy(p), None, None)

    def setParams(self, chipParams: ChipParameters) -> None:
        if chipParams.chipName != self._chipParameters.chipName:
            raise ValueError("Chip mismatch")
        for p in chipParams._params:
            if p.defined:
                super().param(p.name).set_value(p.evalf(), force = True)
