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
from __future__ import annotations  # 3.11: replace with Self

from numbers import Number
from typing import Iterable

class CoherentState:
    """
    This class describes a coherent state, which corresponds to a laser input into one or several modes.

    The power of the laser on one mode is abs(state[m]) ** 2.
    Its relative phase for each mode is given by the complex phase of the amplitude
    """

    def __init__(self, amplitudes: Iterable[Number] = None):
        self._amplitudes: list[complex] = []
        if amplitudes is not None:
            for ampli in amplitudes:
                self._amplitudes.append(complex(ampli))

    def __repr__(self):
        return "|" + ", ".join(map(lambda nb: str(nb)[1:-1], self._amplitudes)) + ">"

    @property
    def m(self) -> int:
        """
        :return: The number of modes
        """
        return len(self._amplitudes)

    def __len__(self) -> int:
        return len(self._amplitudes)

    def __eq__(self, other: CoherentState) -> bool:
        return self._amplitudes == other._amplitudes

    def __ne__(self, other: CoherentState) -> bool:
        return not self != other

    def __getitem__(self, item: int) -> complex:
        return self._amplitudes[item]

    def __mul__(self, other: CoherentState) -> CoherentState:
        return CoherentState(self._amplitudes + other._amplitudes)

    def __pow__(self, power: int) -> CoherentState:
        return CoherentState(self._amplitudes * power)

    def __iter__(self):
        return iter(self._amplitudes)

    def merge(self, other: CoherentState) -> CoherentState:
        assert self.m == other.m, f"Inconsistent number of modes (received {self.m} and {other.m})."
        return CoherentState(self_ampli + other_ampli for self_ampli, other_ampli in zip(self, other))

    def get_power(self) -> list[float]:
        """
        :return: A list of the measurable per-mode power (in the unit used to define the state initially)
        """
        return list(ampli.imag ** 2 + ampli.real ** 2 for ampli in self)
