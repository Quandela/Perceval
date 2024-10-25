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
from enum import Enum

from .abstract_component import AComponent
from .linear_circuit import Circuit
from .unitary_components import BS, PERM
from perceval.utils import BasicState, BSDistribution
from perceval.utils.logging import get_logger, channel


class DetectorType(Enum):
    PNR = 0
    Threshold = 1
    PPNR = 2


class IDetector(AComponent, ABC):

    def __init__(self):
        super().__init__(1)

    @property
    @abstractmethod
    def type(self) -> DetectorType:
        """
        Returns the detector type
        """

    @abstractmethod
    def detect(self, theoretical_photons: int) -> BSDistribution or BasicState:
        """
        Returns a one mode Fock state distribution out of a theoretical photon count coming in the detector
        """


class BSLayeredPPNR(IDetector):
    """
    BSLayeredPPNR implements Pseudo Photon Number Resolving detection using layers of beam splitter plugged on
    2**(number of layers) threshold detectors.

    :param bs_layers: Number of beam splitter layers. Adding more layers enabled to detect
    """

    def __init__(self, bs_layers: int, reflectivity: float = 0.5):
        assert isinstance(bs_layers, int) and bs_layers > 0,\
            "Beam-splitter layers have to be a stricly positive integer"
        assert 0 <= reflectivity <= 1, f"Reflectivity must be between 0 and 1 (got {reflectivity})"
        super().__init__()
        self.name = f"BS-PPNR{bs_layers}"
        self._layers = bs_layers
        self._r = reflectivity

    @property
    def type(self) -> DetectorType:
        return DetectorType.PPNR

    def create_circuit(self) -> Circuit:
        """
        Creates the circuit to simulate PPNR with threshold detectors
        """
        ppnr_circuit = Circuit(2 ** self._layers)
        for l in range(self._layers):
            perm_vector = list(range(0, 2**(l+1)-1, 2)) + list(range(1, 2**(l+1)-1, 2))
            if len(perm_vector) > 1:
                ppnr_circuit.add(0, PERM(perm_vector))
            for m in range(0, 2**(l+1), 2):
                ppnr_circuit.add(m, BS(BS.r_to_theta(self._r)))
        return ppnr_circuit

    def detect(self, theoretical_photons: int) -> BSDistribution or BasicState:
        if theoretical_photons < 2:
            return BasicState([theoretical_photons])

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
        return output


class Detector(IDetector):
    def __init__(self, p_multiphoton_detection: float = 1):
        super().__init__()
        assert 0 <= p_multiphoton_detection <= 1,\
            f"A probability must be within 0 and 1 (got {p_multiphoton_detection})"
        self._pmd = p_multiphoton_detection
        if self.type == DetectorType.PPNR:
            get_logger().error("Generic PPNR was not implemented yet and will behave like a threshold detector for now",
                               channel.user)

    @staticmethod
    def threshold():
        d = Detector(0)
        d.name = "Threshold"
        return d

    @staticmethod
    def pnr():
        d = Detector()
        d.name = "PNR"
        return d

    @staticmethod
    def ppnr(p: float):
        d = Detector(p)
        d.name = f"PPNR"
        return d

    @property
    def type(self) -> DetectorType:
        if self._pmd == 0:
            return DetectorType.Threshold
        elif self._pmd == 1:
            return DetectorType.PNR
        return DetectorType.PPNR

    def detect(self, theoretical_photons: int) -> BSDistribution or BasicState:
        if theoretical_photons < 2 or self.type == DetectorType.PNR:
            return BasicState([theoretical_photons])
        # Adjust the model to treat the PPNR case here
        return BasicState([1])
