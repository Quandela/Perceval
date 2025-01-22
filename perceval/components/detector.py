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
from abc import ABC, abstractmethod
from enum import Enum
from functools import cache

from .abstract_component import AComponent
from .linear_circuit import Circuit
from .unitary_components import BS, PERM
from perceval.utils import BasicState, BSDistribution
from perceval.utils.logging import channel, get_logger


class DetectionType(Enum):
    """Type of photon detection.
    """
    PNR = 0
    Threshold = 1
    PPNR = 2
    Mixed = 3


DetectionType.PNR.__doc__ = "Photon Number Resolving (perfect detection)"
DetectionType.Threshold.__doc__ = "Threshold detection (detects 1 photon at most)"
DetectionType.PPNR.__doc__ = "Pseudo PNR"
DetectionType.Mixed.__doc__ = "Multiple DetectionType"


class IDetector(AComponent, ABC):

    def __init__(self):
        super().__init__(1)

    @property
    @abstractmethod
    def type(self) -> DetectionType:
        """
        Returns the detector type.
        """

    @abstractmethod
    def detect(self, theoretical_photons: int) -> BSDistribution | BasicState:
        """
        Returns a one mode Fock state or distribution out of a theoretical photon count hitting the detector.

        :param theoretical_photons: Number of photons hitting the detector simultaneously.
        :return: The resulting measured state or distribution of all possible measurements.
        """

    def copy(self, subs=None) -> IDetector:
        return copy.copy(self)

    @property
    @abstractmethod
    def max_detections(self):
        pass


class BSLayeredPPNR(IDetector):
    """
    BSLayeredPPNR implements Pseudo Photon Number Resolving detection using layers of beam splitter plugged on
    :math:`2^{(number\ of\ layers)}` threshold detectors.

    :param bs_layers: Number of beam splitter layers.
                    Adding more layers improves the probability to detect multiple photons.
    :param reflectivity: Reflectivity of the beam splitters used to split photons. (defaults to 0.5)
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
    def max_detections(self) -> int:
        """Maximum number of detected photons"""
        return 2 ** self._layers

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
        Creates the beam splitter layered circuit to simulate PPNR with threshold detectors.
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
    Interleaved detector model
    --------------------------

    Such a detector is made of one or multiple wires, each able to simultaneously detect a photon.
    Each photon hitting the detector is absorbed randomly by one of the wires.
    When photons hit the same wire, only one is detected.
    When they hit different wires, all are detected.

    The :code:`detect` method takes the number of wires into account to simulate the detection probability for each case.
    Having 1 wire makes the detector threshold, whereas having an infinity of them makes the detector perfectly PNR.

    :param n_wires: Number of detecting wires in the interleaved detector. (defaults to infinity)
    :param max_detections: Max number of photons the user is willing to read. The `|max_detection>` state would then mean "max_detection or more photons were detected". (defaults to None)

    See :code:`pnr()`, :code:`threshold()` and :code:`ppnr(n_wires, max_detections)` static methods for easy detector initialization.

    Example:

    >>> from perceval.components import Detector
    >>> ppnr_detector = Detector.ppnr(5, 2)  # Create a 5-wires interleaved detector, able to detect 1 or 2+ photons
    >>> print(ppnr_detector.detect(3))       # and simulate the outcome of 3 photons hitting it at once
    {
      |1>: 0.04
      |2>: 0.96
    }
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

    @property
    def max_detections(self) -> int:
        """Maximum number of detected photons (None for infinity)"""
        return self._max

    @staticmethod
    def threshold() -> Detector:
        """Builds a threshold detector."""
        d = Detector(1)
        d.name = "Threshold"
        return d

    @staticmethod
    def pnr() -> Detector:
        """Builds a perfect photon number resolving (PNR) detector."""
        d = Detector()
        d.name = "PNR"
        return d

    @staticmethod
    def ppnr(n_wires: int, max_detections: int = None) -> Detector:
        """Builds an interleaved pseudo-PNR detector."""
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
    """Computes a global detection type from a given list of detectors.

    :param detectors: List of detectors (None is treated as PNR).
    :return:
        * :code:`DetectionType.PNR` if all detectors are PNR or not set.
        * :code:`DetectionType.Threshold` if all detectors are threshold.
        * :code:`DetectionType.PPNR` if all detectors are PPNR.
        * else :code:`DetectionType.Mixed`.
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


def check_heralds_detectors(heralds: dict[int, int] | None, detectors: list[IDetector | None] | None) -> bool:
    """
    Check that heralds are compatible with the given detectors.

    :param heralds: A dictionary mapping herald mode to its value.
    :param detectors: List of detectors (None is treated as PNR).

    :return: True if the maximum value for all detectors is bigger than the expected herald value
    """
    if heralds and detectors:
        for k, v in heralds.items():
            detector = detectors[k]
            if detector:
                max_val = detector.max_detections
                if max_val is not None and max_val < v:
                    get_logger().warn(f"Incompatible heralds and detectors on mode {k}", channel.user)
                    return False

    return True
