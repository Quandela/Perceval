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

from perceval.components import ACircuit
from perceval.utils import Annotation, BasicState, BSDistribution, StateVector, SVDistribution
from perceval.backends._abstract_backends import AProbAmpliBackend

from abc import ABC, abstractmethod
from copy import copy
from multipledispatch import dispatch
from typing import Set, Union


def _to_bsd(sv: StateVector) -> BSDistribution:
    res = BSDistribution()
    for state, pa in sv.items():
        state.clear_annotations()
        res[state] += abs(pa) ** 2
    return res


def _inject_annotation(sv: StateVector, annotation: Annotation) -> StateVector:
    res_sv = copy(sv)
    if len(annotation):
        for s in res_sv:
            s.inject_annotation(annotation)
    return res_sv


def _merge_sv(sv1: StateVector, sv2: StateVector) -> StateVector:
    res = StateVector()
    for s1, pa1 in sv1.items():
        for s2, pa2 in sv2.items():
            res[s1.merge(s2)] = pa1*pa2
    return res


def _annot_state_mapping(bs_with_annots: BasicState):
    bs_list = bs_with_annots.separate_state(keep_annotations=True)
    mapping = {}
    for bs in bs_list:
        annot = bs.get_photon_annotation(0)
        bs.clear_annotations()
        mapping[annot] = bs
    return mapping


class Simulator:

    def __init__(self, backend: AProbAmpliBackend):
        self._backend = backend
        self._invalidate_cache()

    def set_circuit(self, circuit: ACircuit):
        """Set a circuit for simulation.
        :param circuit: a unitary circuit without polarized components
        """
        self._invalidate_cache()
        self._backend.set_circuit(circuit)

    @dispatch(BasicState, BasicState)
    def prob_amplitude(self, input_state: BasicState, output_state: BasicState) -> complex:
        """Compute the probability amplitude of an output fock state versus an input fock state.
        :param input_state: A fock state with or without photon annotations
        :param output_state: A fock state with or without photon annotations. If the input state holds annotations,
            the output state must hold the same ones, otherwise the computed probability amplitude is 0.

            >>> simulator.set_circuit(Circuit(1))  # One mode identity
            >>> simulator.prob_amplitude(BasicState('|{_:0}>'), BasicState('|{_:1}>'))
            0
            >>> simulator.prob_amplitude(BasicState('|{_:0}>'), BasicState('|{_:0}>'))
            1

        :return: The complex probability amplitude
        """
        if input_state.n == 0:
            return complex(1) if output_state.n == 0 else complex(0)
        input_map = _annot_state_mapping(input_state)
        output_map = _annot_state_mapping(output_state)
        if len(input_map) != len(output_map):
            return complex(0)
        probampli = 1
        for annot, in_s in input_map.items():
            if annot not in output_map:
                return complex(0)
            self._backend.set_input_state(in_s)
            probampli *= self._backend.prob_amplitude(output_map[annot])
        return probampli

    @dispatch(StateVector, BasicState)
    def prob_amplitude(self, input_state: StateVector, output_state: BasicState) -> complex:
        result = complex(0)
        for state, pa in input_state.items():
            result += self.prob_amplitude(state, output_state) * pa
        return result

    @dispatch(BasicState, BasicState)
    def probability(self, input_state: BasicState, output_state: BasicState) -> float:
        """Compute the probability of an output fock state versus an input fock state, simulating a measure.
        :param input_state: A fock state with or without photon annotations
        :param output_state: A fock state, annotations are ignored
        :return: The probability (float between 0 and 1)
        """
        if input_state.n == 0:
            return 1 if output_state.n == 0 else 0
        input_list = input_state.separate_state(keep_annotations=False)
        result = 0
        for p_output_state in output_state.partition(
                [input_state.n for input_state in input_list]):
            prob = 1
            for i_state, o_state in zip(input_list, p_output_state):
                self._backend.set_input_state(i_state)
                prob *= self._backend.probability(o_state)
            result += prob
        return result

    @dispatch(StateVector, BasicState)
    def probability(self, input_state: StateVector, output_state: BasicState) -> float:
        output_state.clear_annotations()
        sv_out = self.evolve(input_state)  # This is not as optimized as it could be
        result = 0
        for state, pa in sv_out.items():
            state.clear_annotations()
            if state == output_state:
                result += abs(pa) ** 2
        return result

    def _invalidate_cache(self):
        self._cache = {}
        self.DEBUG_evolve_count = 0
        self.DEBUG_merge_count = 0

    def _evolve_cache(self, input_list: Set[BasicState]):
        for state in input_list:
            if state not in self._cache:
                self._backend.set_input_state(state)
                self._cache[state] = self._backend.evolve()
                self.DEBUG_evolve_count += 1

    def _merge_probability_dist(self, input_list):
        results = BSDistribution()
        for input_state in input_list:
            results = BSDistribution.tensor_product(results, _to_bsd(self._cache[input_state]), merge_modes=True)
            self.DEBUG_merge_count += 1
        return results

    @dispatch(BasicState)
    def probs(self, input_state: BasicState) -> BSDistribution:
        input_list = input_state.separate_state(keep_annotations=False)
        self._evolve_cache(set(input_list))
        return self._merge_probability_dist(input_list)

    @dispatch(StateVector)
    def probs(self, input_state: StateVector) -> BSDistribution:
        if len(input_state) == 1:
            return self.probs(input_state[0])
        return _to_bsd(self.evolve(input_state))

    @dispatch(SVDistribution)
    def probs(self, input_state: SVDistribution):
        raise NotImplementedError()

    def evolve(self, input_state: Union[BasicState, StateVector]) -> StateVector:
        if not isinstance(input_state, StateVector):
            input_state = StateVector(input_state)

        # Decay input to a list of basic states without annotations and evolve each of them
        decomposed_input = [(pa, st.separate_state(keep_annotations=True)) for st, pa in input_state.items()]
        input_list = [copy(state) for t in decomposed_input for state in t[1]]
        for state in input_list:
            state.clear_annotations()
        self._evolve_cache(set(input_list))

        result_sv = StateVector()
        for probampli, instate_list in decomposed_input:
            reslist = []
            for in_s in instate_list:
                annotation = in_s.get_photon_annotation(0)
                in_s.clear_annotations()
                reslist.append(_inject_annotation(self._cache[in_s], annotation))

            # Recombine results for one basic state input
            evolved_in_s = reslist.pop(0)
            for sv in reslist:
                evolved_in_s = _merge_sv(evolved_in_s, sv)
                self.DEBUG_merge_count += 1
            result_sv += evolved_in_s * probampli

        result_sv.normalize()
        return result_sv


class ASimulatorDecorator(ABC):
    def __init__(self, simulator: Simulator):
        self._simulator = simulator

    @abstractmethod
    def _prepare_input(self, input_state):
        pass

    @abstractmethod
    def _prepare_circuit(self, circuit) -> ACircuit:
        pass

    @abstractmethod
    def _postprocess_results(self, results):
        pass

    def set_circuit(self, circuit):
        self._simulator.set_circuit(self._prepare_circuit(circuit))

    def probs(self, input_state):
        results = self._simulator.probs(self._prepare_input(input_state))
        return self._postprocess_results(results)

    def evolve(self, input_state):
        results = self._simulator.evolve(self._prepare_input(input_state))
        return self._postprocess_results(results)
