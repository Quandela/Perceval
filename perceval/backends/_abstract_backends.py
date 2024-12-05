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
import exqalibur as xq

from perceval.components import ACircuit
from perceval.utils import BasicState, BSDistribution, BSSamples, allstate_iterator, StateVector
from perceval.utils.logging import deprecated


class ABackend(ABC):
    def __init__(self):
        self._circuit = None
        self._umat = None
        self._input_state = None

    def set_circuit(self, circuit: ACircuit):
        if circuit.requires_polarization:
            raise RuntimeError("Circuit must not contain polarized components")
        self._input_state = None
        self._circuit = circuit
        self._umat = circuit.compute_unitary()

    def set_input_state(self, input_state: BasicState):
        self._check_state(input_state)
        self._input_state = input_state

    def _check_state(self, state: BasicState):
        assert self._circuit is not None, 'Circuit must be set before the input state'
        assert self._circuit.m == state.m, f'Circuit({self._circuit.m}) and state({state.m}) size mismatch'
        assert not state.has_annotations, 'State should be composed of indistinguishable photons only'

    @property
    @abstractmethod
    def name(self) -> str:
        """Returns the back-end name as a string"""


class ASamplingBackend(ABackend):
    @abstractmethod
    def sample(self) -> BasicState:
        """Request one sample from the circuit given an input state"""

    @abstractmethod
    def samples(self, count: int) -> BSSamples:
        """Request samples from the circuit given an input state"""


class AStrongSimulationBackend(ABackend):

    def __init__(self):
        super().__init__()
        self._cache_iterator: dict = dict()
        self._masks_str: list[str] | None = None
        self._mask: xq.FSMask | None = None

    def set_mask(self, masks: str | list[str]):
        """
        Sets new masks, replacing the former ones if they exist.
        Masks are useful to limit strong simulation to only a part of the Fock space, ultimately saving memory and
        computation time.

        :param masks: Can be a mask or a list of masks. Each mask is expressed as a string where each character is a
            condition on one mode. Digits are fixing the number of photons whereas spaces or "*" are accepting any
            number of detections. e.g. using "****00" as a mask limits the simulation to output states ending in two
            empty modes.
        """
        self.clear_mask()
        if isinstance(masks, str):
            masks = [masks]
        mask_length = len(masks[0])
        for m in masks:
            m = m.replace("*", " ")
            assert len(m) == mask_length, "Inconsistent mask lengths"
        self._masks_str = masks
        self._init_mask()

    def _init_mask(self):
        if self._masks_str is not None and self._input_state is not None:
            instate = self._input_state
            assert len(self._masks_str[0]) == instate.m, "Mask and input state lengths have to be the same"
            self._mask = xq.FSMask(instate.m, instate.n, self._masks_str)

    def clear_mask(self):
        """
        Removes any pre-existing mask
        """
        self._masks_str = None
        self._mask = None
        self.clear_iterator_cache()

    def set_input_state(self, input_state: BasicState):
        """
        Sets an input state for the simulation. This state has to be a Fock state without annotations.
        """
        super().set_input_state(input_state)
        self._init_mask()

    def _get_iterator(self, input_state: BasicState):
        n_photons = input_state.n

        if n_photons not in self._cache_iterator.keys():
            self._cache_iterator[n_photons] = tuple(allstate_iterator(input_state, self._mask))

        return self._cache_iterator[n_photons]

    def clear_iterator_cache(self):
        self._cache_iterator = dict()

    def set_circuit(self, circuit: ACircuit):
        """
        Sets the circuit to simulate. This circuit must not contain polarized components (use PolarizationSimulator
        instead, if required).
        """
        if self._circuit and circuit.m != self._circuit:
            self.clear_iterator_cache()
        super().set_circuit(circuit)

    @abstractmethod
    def prob_amplitude(self, output_state: BasicState) -> complex:
        pass

    def probability(self, output_state: BasicState) -> float:
        return abs(self.prob_amplitude(output_state)) ** 2

    def all_prob(self, input_state: BasicState = None) -> list[float]:
        if input_state is not None:
            self.set_input_state(input_state)
        results = []
        for output_state in self._get_iterator(self._input_state):
            results.append(self.probability(output_state))
        return results

    def prob_distribution(self) -> BSDistribution:
        bsd = BSDistribution()
        for output_state in self._get_iterator(self._input_state):
            bsd.add(output_state, self.probability(output_state))
        return bsd

    def evolve(self) -> StateVector:
        res = StateVector()
        for output_state in self._get_iterator(self._input_state):
            res += output_state * self.prob_amplitude(output_state)
        res.normalize()
        return res


class AProbAmpliBackend(AStrongSimulationBackend, ABC):
    """Deprecated: this class was renamed to AStrongSimulationBackend"""

    @deprecated(version="0.12.0", reason="AProbAmpliBackend was renamed to AStrongSimulationBackend")
    def __init__(self):
        super().__init__()
