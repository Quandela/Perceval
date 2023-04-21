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
from perceval.utils import BasicState, BSDistribution, StateVector, SVDistribution
from perceval.backends._abstract_backends import AProbAmpliBackend

from abc import ABC, abstractmethod
from copy import copy
from multipledispatch import dispatch
from typing import Set, Union


class Simulator:

    def __init__(self, backend: AProbAmpliBackend):
        self._backend = backend
        self._pdists = []
        self.DEBUG_computation_count = 0

    def set_circuit(self, circuit: ACircuit):
        self._backend.set_circuit(circuit)

    # def prob_amplitude(self, input_state: BasicState, output_state: BasicState):
    #     if input_state.n == 0:
    #         return complex(1) if output_state.n == 0 else complex(0)
    #     input_list = input_state.separate_state()

    def _cache_output_dist(self, input_set: Set[BasicState]):
        self.DEBUG_computation_count = len(input_set)
        self._pdists = []
        self._iset = list(input_set)
        for input_state in input_set:
            self._backend.set_input_state(input_state)
            self._pdists.append(self._backend.prob_distribution())

    # def _cache_evolve(self, input_set: Set[BasicState]):
    #     self.DEBUG_computation_count = len(input_set)
    #     self._padists = []
    #     self._iset = list(input_set)
    #     for input_state in input_set:
    #         self._backend.set_input_state(input_state)
    #         self._padists.append(self._backend.())

    def _merge_probability_dist(self, input_list):
        results = BSDistribution()
        for input_state in input_list:
            idx = self._iset.index(input_state)
            results = BSDistribution.tensor_product(results, self._pdists[idx], merge_modes=True)
        return results

    @dispatch(BasicState)
    def probs(self, input_state: BasicState) -> BSDistribution:
        input_list = input_state.separate_state(keep_annotations=False)
        self._cache_output_dist(set(input_list))
        return self._merge_probability_dist(input_list)

    @dispatch(StateVector)
    def probs(self, input_state: StateVector) -> BSDistribution:
        if len(input_state) == 1:
            return self.probs(input_state[0])
        input_set = set()
        for fock_state in input_state:
            input_set.union(fock_state.separate_state())
        self._cache_output_pa_dist(input_set)
        for fs, amp in input_state.items():
            pass

    def _merge_sv(self, sv1, sv2) -> StateVector:
        res = StateVector()
        for s1, pa1 in sv1.items():
            for s2, pa2 in sv2.items():
                res[s1.merge(s2)] = pa1*pa2
        return res

    def evolve(self, input_state: Union[BasicState, StateVector]) -> StateVector:
        if not isinstance(input_state, StateVector):
            input_state = StateVector(input_state)

        input_list = [(pa, st.separate_state()) for st, pa in input_state.items()]
        input_set = [copy(state) for t in input_list for state in t[1]]
        for state in input_set:
            state.clear_annotations()
        input_set = list(set(input_set))  # Remove duplicates
        sv_cache = []
        for inps in input_set:
            self._backend.set_input_state(inps)
            sv_cache.append(self._backend.evolve())
        result_sv = StateVector()
        final_list = []
        for in_s_list in input_list:
            reslist = []
            for in_s in in_s_list[1]:
                annotation = in_s.get_photon_annotation(0)
                in_s.clear_annotations()
                sv = copy(sv_cache[input_set.index(in_s)])
                # Re-inject annotation
                if len(annotation):
                    for s in sv:
                        s.inject_annotation(annotation)
                reslist.append(sv)

            # Recombine results for one basic state input
            res = reslist.pop(0)
            for sv in reslist:
                res = self._merge_sv(res, sv)

            result_sv += res * in_s_list[0]
            final_list.append(res)

        result_sv.normalize()
        return result_sv

    @dispatch(SVDistribution)
    def probs(self, input_state: SVDistribution):
        raise NotImplementedError()


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
