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
from typing import Iterable

import exqalibur as xq

from perceval.components import ACircuit
from perceval.utils import BasicState, FockState, BSDistribution, BSSamples, StateVector, allstate_iterator, global_params


class ABackend(ABC):
    def __init__(self):
        self._circuit = None
        self._umat = None
        self._input_state = None

    def set_circuit(self, circuit: ACircuit):
        """
        Sets the circuit to simulate. This circuit must not contain polarized components (use PolarizationSimulator
        instead, if required).
        """
        if circuit.requires_polarization:
            raise RuntimeError("Circuit must not contain polarized components")
        self._input_state = None
        self._circuit = circuit
        self._umat = circuit.compute_unitary()

    def set_input_state(self, input_state: FockState):
        """
        Sets an input state for the simulation. This state has to be a Fock state without annotations.
        """
        self._check_state(input_state)
        self._input_state = input_state

    def _check_state(self, state: FockState):
        assert self._circuit is not None, 'Circuit must be set before the input state'
        assert self._circuit.m == state.m, f'Circuit({self._circuit.m}) and state({state.m}) size mismatch'

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


class _StateProbIterator(Iterable[tuple[FockState, float]]):

    def __init__(self, states: Iterable[FockState], probs: Iterable[float]):
        self.states = states
        self.probs = probs

    def __iter__(self):
        return zip(self.states, self.probs)


class AStrongSimulationBackend(ABackend):

    def __init__(self):
        super().__init__()
        self._mask_n = None
        self._cache_iterator: dict = dict()
        self._masks_str: list[str] | None = None
        self._mask: xq.FSMask | None = None

    def set_mask(self, masks: str | list[str], n = None):
        r"""
        Sets new masks, replacing the former ones if they exist. Clear possible cached data that depend on the mask.
        Masks are useful to limit strong simulation to only a part of the Fock space, ultimately saving memory and
        computation time.

        :param masks: Can be a mask or a list of masks. Each mask is expressed as a string where each character is a
            condition on one mode. Digits are fixing the number of photons whereas spaces or "\*" are accepting any
            number of detections. e.g. using "\*\*\*\*00" as a mask limits the simulation to output states ending in two
            empty modes.
        :param n: The number of photons to instantiate the mask with.
            This corresponds to the total number of photons in your non-separated state.
        """
        self.clear_mask()
        if isinstance(masks, str):
            masks = [masks]
        mask_length = len(masks[0])
        for m in masks:
            m = m.replace("*", " ")
            assert len(m) == mask_length, "Inconsistent mask lengths"
        self._masks_str = masks
        self._mask_n = n
        self._init_mask()

    def _init_mask(self):
        if self._masks_str is not None and self._input_state is not None:
            instate = self._input_state
            assert len(self._masks_str[0]) == instate.m, "Mask and input state lengths have to be the same"
            self._mask = xq.FSMask(instate.m, self._mask_n or instate.n, self._masks_str)

    def clear_mask(self):
        """
        Removes any pre-existing mask
        """
        self._masks_str = None
        self._mask = None
        self._mask_n = None
        self.clear_iterator_cache()

    def set_input_state(self, input_state: FockState):
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
        if self._circuit and circuit.m != self._circuit.m:
            self.clear_iterator_cache()
        super().set_circuit(circuit)

    @abstractmethod
    def prob_amplitude(self, output_state: FockState) -> complex:
        """Computes the probability amplitude for a given output state. The input state and the circuit must already be set"""
        pass

    def probability(self, output_state: FockState) -> float:
        """Computes the probability for a given output state. The input state and the circuit must already be set"""
        return abs(self.prob_amplitude(output_state)) ** 2

    def all_prob(self, input_state: FockState = None) -> list[float]:
        """Computes the list of probabilities of all states (respecting the mask if any was set).
        The order of the states can be retrieved using `allstate_iterator()`"""
        if input_state is not None:
            self.set_input_state(input_state)
        results = []
        for output_state in self._get_iterator(self._input_state):
            results.append(self.probability(output_state))
        return results

    def prob_distribution(self) -> BSDistribution:
        """
        Computes the probability distribution of all states (respecting the mask if any was set)
        under the form of a BSDistribution.
        """
        bsd = BSDistribution()
        for output_state in self._get_iterator(self._input_state):
            bsd.add(output_state, self.probability(output_state))
        return bsd

    def prob_iterator(self, min_p: float = global_params["min_p"]) -> Iterable[tuple[FockState, float]]:
        # DO NOT document for users
        probs = self.all_prob(self._input_state)
        states = [state for i, state in enumerate(self._get_iterator(self._input_state)) if probs[i] > min_p]
        return _StateProbIterator(states, [prob for prob in probs if prob > min_p])

    def evolve(self) -> StateVector:
        """Evolves the input BasicState into a StateVector."""
        res = StateVector()
        for output_state in self._get_iterator(self._input_state):
            res += output_state * self.prob_amplitude(output_state)
        return res
