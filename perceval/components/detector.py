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
from abc import ABC, abstractmethod
from enum import Enum
from functools import cache

from .abstract_component import AComponent
from .linear_circuit import Circuit
from .unitary_components import BS, PERM
from perceval.utils import BasicState, BSDistribution


class DetectionType(Enum):
    PNR = 0
    Threshold = 1
    PPNR = 2
    Mixed = 3


class IDetector(AComponent, ABC):

    def __init__(self):
        super().__init__(1)

    @property
    @abstractmethod
    def type(self) -> DetectionType:
        """
        Returns the detector type
        """

    @abstractmethod
    def detect(self, theoretical_photons: int) -> BSDistribution | BasicState:
        """
        Returns a one mode Fock state or distribution out of a theoretical photon count coming in the detector.

        :param theoretical_photons: Number of photons coming at once into the detector
        :return: The resulting measured state or distribution of all possible measurements
        """


class BSLayeredPPNR(IDetector):
    """
    BSLayeredPPNR implements Pseudo Photon Number Resolving detection using layers of beam splitter plugged on
    2**(number of layers) threshold detectors.

    :param bs_layers: Number of beam splitter layers. Adding more layers enabled to detect
    :param reflectivity: Reflectivity of the beam splitters used to split photons
    """

    def __init__(self, bs_layers: int, reflectivity: float = 0.5):
        assert isinstance(bs_layers, int) and bs_layers > 0,\
            f"Beam-splitter layers have to be a stricly positive integer (got {bs_layers})"
        assert 0 <= reflectivity <= 1, f"Reflectivity must be between 0 and 1 (got {reflectivity})"
        super().__init__()
        self.name = f"BS-PPNR{bs_layers}"
        self._layers = bs_layers
        self._r = reflectivity
        self._cache = {}  # This cache records simulations for a given photon count to speed up computations

    @property
    def type(self) -> DetectionType:
        return DetectionType.PPNR

    def clear_cache(self):
        """
        Detector simulation results are cached in each instance and may consume memory.
        Call this method to empty the cache.
        """
        self._cache = {}

    def create_circuit(self) -> Circuit:
        """
        Creates the beam splitter layered circuit to simulate PPNR with threshold detectors
        """
        ppnr_circuit = Circuit(2 ** self._layers)
        for l in range(self._layers):
            perm_vector = list(range(0, 2**(l+1)-1, 2)) + list(range(1, 2**(l+1)-1, 2))
            if len(perm_vector) > 1:
                ppnr_circuit.add(0, PERM(perm_vector))
            for m in range(0, 2**(l+1), 2):
                ppnr_circuit.add(m, BS(BS.r_to_theta(self._r)))
        return ppnr_circuit

    def detect(self, theoretical_photons: int) -> BSDistribution | BasicState:
        if theoretical_photons < 2:
            return BasicState([theoretical_photons])

        if theoretical_photons in self._cache:
            return self._cache[theoretical_photons]

        from perceval.backends import SLOSBackend
        ppnr_circuit = self.create_circuit()
        slos = SLOSBackend()
        slos.set_circuit(ppnr_circuit)
        slos.set_input_state(BasicState([theoretical_photons] + [0]*(ppnr_circuit.m - 1)))
        dist = slos.prob_distribution()

        output = BSDistribution()
        for state, prob in dist.items():
            state = state.threshold_detection()
            output[BasicState([state.n])] += prob
        self._cache[theoretical_photons] = output
        return output


class Detector(IDetector):
    """
    Interleaved detector class

    Such a detector is made of one or multiple wires, each able to simultaneously detect a photon. The `detect` method
    takes the number of wires into acocunt to simulate the detection probability for each case.
    Having 1 wire makes the detector threshold, whereas having an infinity of them makes the detector perfectly PNR.

    :param n_wires: Number of detecting wires in the interleaved detector (defaults to infinity)
    :param max_detections: Max number of photons the user is willing to read. The |max_detection> state would then mean
                           "max_detection or more photons were detected". (defaults to None)

    See `pnr()`, `threshold()` and `ppnr(n_wires, max_detections)` static methods for easy detector initialization
    """

    def __init__(self, n_wires: int = None, max_detections: int = None):
        super().__init__()
        assert n_wires is None or n_wires > 0, f"A detector requires at least 1 wire (got {n_wires})"
        assert max_detections is None or n_wires is None or max_detections <= n_wires, \
            f"Max detections has to be lower or equal than the number of wires (got {max_detections} > {n_wires} wires)"
        self._wires = n_wires
        self._max = None
        if self._wires is not None:
            self._max = self._wires if max_detections is None else min(max_detections, self._wires)
        self._cache = {}

    @staticmethod
    def threshold():
        """Builds a threshold detector"""
        d = Detector(1)
        d.name = "Threshold"
        return d

    @staticmethod
    def pnr():
        """Builds a perfect photon number resolving (PNR) detector"""
        d = Detector()
        d.name = "PNR"
        return d

    @staticmethod
    def ppnr(n_wires: int, max_detections: int = None):
        """Builds an interleaved pseudo-PNR detector"""
        d = Detector(n_wires, max_detections)
        d.name = f"PPNR"
        return d

    @property
    def type(self) -> DetectionType:
        if self._wires == 1:
            return DetectionType.Threshold
        elif self._wires is None and self._max is None:
            return DetectionType.PNR
        return DetectionType.PPNR

    def detect(self, theoretical_photons: int) -> BSDistribution | BasicState:
        detector_type = self.type
        if theoretical_photons < 2 or detector_type == DetectionType.PNR:
            return BasicState([theoretical_photons])

        if detector_type == DetectionType.Threshold:
            return BasicState([1])

        if theoretical_photons in self._cache:
            return self._cache[theoretical_photons]

        remaining_p = 1
        result = BSDistribution()
        max_detectable = min(self._max, theoretical_photons)
        for i in range(1, max_detectable):
            p_i = self._cond_probability(i, theoretical_photons)
            result.add(BasicState([i]), p_i)
            remaining_p -= p_i
        # The highest detectable photon count gains all the remaining probability
        result.add(BasicState([max_detectable]), remaining_p)
        self._cache[theoretical_photons] = result
        return result

    @cache
    def _cond_probability(self, det: int, nph: int):
        """
        The conditional probability of having `det` detections with `nph` photons on the total number of wires.
        This uses a recurrence formula set to compute each conditional probability from the ones with one less photon.
        Hitting `i` wires with `n` photons is:
        - hitting `i - 1` wires with `n - 1` photons AND hitting a new wire with the nth photon
        OR
        - hitting `i` wires with `n - 1` photons AND hitting one of the wire that were already hit with the nth photon
        """
        if det == 0:
            return 1 if nph == 0 else 0
        if nph < det:
            return 0
        return self._cond_probability(det - 1, nph - 1) * (self._wires - det + 1) / self._wires \
            + self._cond_probability(det, nph - 1) * det / self._wires


def get_detection_type(detectors: list[IDetector]) -> DetectionType:
    """
    Computes a global detection type from a given list of detectors.

    :param detectors: List of detectors (None is treated as PNR)
    :return: PNR if all detectors are PNR or not set
             Threshold if all detectors are threshold
             PPNR if all detectors are PPNR
             else Mixed
    """
    if not detectors:
        return DetectionType.PNR  # To keep previous behavior where not setting any detector would mean PNR
    result = None
    for det in detectors:
        current = DetectionType.PNR if det is None else det.type  # Default is PNR
        if result is None:
            result = current
        elif result != current:
            return DetectionType.Mixed
    return result
