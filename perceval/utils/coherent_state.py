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
from typing import Iterable

from exqalibur import StateVector, FockState


class CoherentState:
    """
    This class describes a coherent state, which corresponds to a laser inputted into one or several modes.

    The power of the laser on one mode is abs(state[m]) ** 2.
    Its relative phase for each mode is given by the complex phase of the amplitude,
    and represents the time phase of the electromagnetic field.
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

    @property
    def n(self) -> int:
        return 1 if self.m > 0 and any(ampli for ampli in self) else 0

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

    @property
    def has_polarization(self) -> bool:
        return False

    @property
    def has_annotations(self) -> bool:
        return False

    def remove_modes(self, modes: list[int]) -> CoherentState:
        """
        :param modes: The list of modes to remove.
        :return: A new state with only the modes that are not in ``modes``
        """
        return CoherentState(ampli for m, ampli in enumerate(self) if m not in modes)

    def set_slice(self, other: CoherentState, start: int, end: int) -> CoherentState:
        """
        :param other: The state to replace part of ``self`` with. Must have :math:`end - start` modes
        :param start: The first mode to replace.
        :param end: The last mode to replace (excluded).
        :return: A new state where the section between start and end is ``other``, and the remaining comes from ``self``.
        """
        assert other.m == end - start, f"Incorrect number of modes (received {other.m}, expected {end - start})."
        assert end <= self.m, "Can't replace modes beyond the number of modes of this state"
        assert start >= 0, "Can't replace negative modes"

        return CoherentState(self[m] if m < start or m >= end else other[m - start] for m in range(self.m))

    def to_power(self) -> list[float]:
        """
        :return: A list of the measurable per-mode power (in the unit used to define the state initially)
        """
        return list(ampli.imag ** 2 + ampli.real ** 2 for ampli in self)

    def to_state_vector(self) -> StateVector:
        """
        :return: An equivalent state vector composed of superposed 1-photon states
        """
        res = StateVector()
        fs_list = [0] * self.m
        for m, ampli in enumerate(self):
            if ampli:
                fs_list[m] = 1
                res += ampli * FockState(fs_list)
                fs_list[m] = 0

        if not len(res):
            res += FockState(fs_list)

        return res
