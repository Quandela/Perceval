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
from packaging.version import Version
from perceval.components.abstract_component import AParametrizedComponent
from perceval.utils.parameter import Parameter

class ChipID:
    def __init__(self, name: str, version: Version | None = None):
        self._name = name
        self._version = version

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> Version:
        return self._version


class ChipParameters:
    def __init__(self, chipId: ChipID, values: list[Parameter] = None):
        self._chipId = chipId
        self._values = values or []

    def get_values(self) -> list[float]:
        result = None
        if all(p.defined for p in self._values):
            result = []
            for p in self._values:
                result.append(p.evalf())
        return result

    def set_values(self, val: list[float]) -> None:
        if len(val) != len(self._values):
            raise "invalid parameters list"
        for p, v in zip(self._values, val):
            p.set_value(val, force = True)

    def need_compilation(self, other: ChipParameters) -> bool:
        if self._chipId.name != other._chipId.name:
            raise ValueError("Chip mismatch")
        if not (self._chipId.version and other._chipId._version):
            return False

        # Consider checking closeness of values
        return self._chipId.version != other._chipId._version


class CompiledCircuit(AParametrizedComponent):
    def __init__(self, m: int, chipParameters: ChipParameters):
        super().__init__(m, chipParameters.chipId.name)
        self._chipParameters = chipParameters

    def setParams(self, chipParams: ChipParameters) -> None:
        if chipParams._chipId != self._chipParameters._chipId:
            raise ValueError("Chip mismatch")
        for p in chipParams._values:
            super.param(p.name).set_value(p.evalf(), force = True)
