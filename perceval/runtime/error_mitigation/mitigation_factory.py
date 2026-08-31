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

from typing import Type
from enum import IntEnum

from .abstract_mitigation import AbstractMitigation
from .compilation_averaging import CompilationAveraging
from .photon_recycling import PhotonRecycling
from .distinguishable_photon_mitigation import DistinguishablePhotonMitigation
from .detector_balancing import DetectorBalancing


class MitigationLevel(IntEnum):
    none = 0
    low = 1
    medium = 2
    high = 3


class MitigationFactory:

    MITIGATION_ORDER: list[Type[AbstractMitigation]] = [
        CompilationAveraging,
        DistinguishablePhotonMitigation,
        PhotonRecycling,  # Never set by the default levels, but can be added by hand by the user
        DetectorBalancing,
    ]

    def __init__(self, level: MitigationLevel = MitigationLevel.low):
        self._mitigations: list[AbstractMitigation | None] = []
        self.reset_to_level(level)

    def build(self) -> list[AbstractMitigation]:
        return [mitigation for mitigation in self._mitigations if mitigation is not None]

    def _set_mitigation(self, mitigation_type: Type[AbstractMitigation], mitigation: AbstractMitigation | None):
        idx = self.MITIGATION_ORDER.index(mitigation_type)
        self._mitigations[idx] = mitigation

    def set_custom_mitigation(self, mitigation: AbstractMitigation):
        if type(mitigation) not in self.MITIGATION_ORDER:
            raise ValueError(f"Unknown mitigation type: {type(mitigation).__name__}. "
                             "If this is not an error, "
                             "add it to MitigationFactory.MITIGATION_ORDER and make a new instance to make it work")
        self._set_mitigation(type(mitigation), mitigation)

    def set_compilation_averaging(self, repetitions: int | None, starting_seed: int = None):
        mitigation = CompilationAveraging(repetitions, starting_seed) if repetitions is not None else None
        self._set_mitigation(CompilationAveraging, mitigation)

    def set_distinguishable_photon_mitigation(self, order: int | dict[int, int] | None):
        mitigation = DistinguishablePhotonMitigation(order) if order is not None else None
        self._set_mitigation(DistinguishablePhotonMitigation, mitigation)

    def set_photon_recycling(self, use_it: bool = True):
        mitigation = PhotonRecycling() if use_it else None
        self._set_mitigation(PhotonRecycling, mitigation)

    def set_detector_balancing(self, use_it: bool = True):
        mitigation = DetectorBalancing() if use_it else None
        self._set_mitigation(DetectorBalancing, mitigation)

    def reset_to_level(self, level: MitigationLevel) -> None:
        self._remove_mitigations()
        match level:
            case MitigationLevel.low:
                self._set_low_level()
            case MitigationLevel.medium:
                self._set_medium_level()
            case MitigationLevel.high:
                self._set_high_level()
            case MitigationLevel.none:
                pass
            case _:
                raise ValueError(f"Unknown mitigation level: {level}")

    def _remove_mitigations(self) -> None:
        self._mitigations = [None] * len(self.MITIGATION_ORDER)

    def _set_low_level(self) -> None:
        self.set_detector_balancing()

    def _set_medium_level(self) -> None:
        self._set_low_level()
        self.set_distinguishable_photon_mitigation(1)
        self.set_compilation_averaging(3)

    def _set_high_level(self) -> None:
        self._set_low_level()
        self.set_distinguishable_photon_mitigation(3)
        self.set_compilation_averaging(5)
