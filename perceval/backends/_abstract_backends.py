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

from perceval.components import ACircuit
from perceval.utils import BasicState, BSDistribution, allstate_iterator, StateVector


class ABackend(ABC):
    def __init__(self):
        self._circuit = None
        self._umat = None
        self._input_state = None

    def set_circuit(self, circuit: ACircuit):
        assert not circuit.requires_polarization, "Circuit must not contain polarized components"
        self._input_state = None
        self._circuit = circuit
        self._umat = circuit.compute_unitary()

    def set_input_state(self, input_state: BasicState):
        self._check_state(input_state)
        self._input_state = input_state

    def _check_state(self, state: BasicState):
        assert self._circuit.m == state.m, f'Circuit({self._circuit.m}) and state({state.m}) size mismatch'
        assert not state.has_annotations, 'State should be composed of indistinguishable photons only'

    @property
    @abstractmethod
    def name(self) -> str:
        """Each backend has to expose its name as a string"""

    @staticmethod
    @abstractmethod
    def preferred_command() -> str:
        pass


class ASamplingBackend(ABackend):
    @abstractmethod
    def sample(self):
        """Request samples from the circuit given an input state"""

    @staticmethod
    def preferred_command() -> str:
        return "sample"


class AProbAmpliBackend(ABackend):
    @abstractmethod
    def prob_amplitude(self, output_state: BasicState) -> complex:
        pass

    def probability(self, output_state: BasicState) -> float:
        return abs(self.prob_amplitude(output_state)) ** 2

    def prob_distribution(self) -> BSDistribution:
        bsd = BSDistribution()
        for output_state in allstate_iterator(self._input_state):
            bsd.add(output_state, self.probability(output_state))
        return bsd

    def evolve(self) -> StateVector:
        res = StateVector()
        for output_state in allstate_iterator(self._input_state):
            res[output_state] = self.prob_amplitude(output_state)
        res.normalize()
        return res

    @staticmethod
    def preferred_command() -> str:
        return "prob_amplitude"
