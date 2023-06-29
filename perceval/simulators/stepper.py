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
from collections import defaultdict
from typing import List, Union, Callable, Dict
import copy

from perceval.utils import StateVector, BasicState, BSDistribution, SVDistribution, allstate_iterator
from perceval.components import ACircuit
from perceval.backends import AProbAmpliBackend, BACKEND_LIST
from .simulator_interface import ISimulator
from ._simulator_utils import _to_bsd


class Stepper(ISimulator):
    """
    Step-by-step circuit propagation algorithm, main usage is on a circuit, but could work in degraded mode
    on a list of components [(r, comp)].
    """
    def __init__(self, backend: AProbAmpliBackend = None):
        self._out = None
        self._backend = backend
        if backend is None:
            self._backend = BACKEND_LIST['SLOS']()
        self._min_detected_photons = 0
        self._clear_cache()
        self._C = None

    def _clear_cache(self):
        self._result_dict = defaultdict(lambda: {'_set': set()})
        self._compiled_input = None

    def set_circuit(self, circuit: ACircuit):
        self._C = circuit
        self._clear_cache()

    def set_min_detected_photon_filter(self, value: int):
        self._min_detected_photons = value

    def apply(self, sv: StateVector, r: List[int], c: ACircuit) -> StateVector:
        """Apply a circuit on a StateVector generating another StateVector
        :param sv: input StateVector
        :param r: range of port for the circuit corresponding to StateVector position
        :param c: a circuit
        :return: evolved StateVector
        """
        min_r = r[0]
        max_r = r[-1] + 1
        key = c.describe()  # Can't use c; two identical pieces aren't considered equal if they aren't at the same place
        # build list of never visited fockstates corresponding to subspace [min_r:max_r]
        sub_input_state = {sliced_state for state in sv
                           for sliced_state in (state[min_r:max_r],)
                           if sliced_state not in self._result_dict[key]['_set']
                           and state[:self._C.m].n >= self._min_detected_photons}
        # get circuit probability for these input_states
        if sub_input_state:
            self._backend.set_circuit(c)
            mapping_input_output = {}
            for input_state in sub_input_state:
                self._backend.set_input_state(input_state)
                mapping_input_output[input_state] = {output_state: self._backend.prob_amplitude(output_state)
                                                     for output_state in allstate_iterator(input_state)}
            self._result_dict[key].update(mapping_input_output)  # Union of the dictionaries
            self._result_dict[key]['_set'] |= sub_input_state  # Union of sets
        # now rebuild the new state vector
        nsv = StateVector()
        # May be faster in c++ (impossible to use comprehension here due to successive additions)
        for state in sv:
            if state[:self._C.m].n >= self._min_detected_photons:  # Useless to compute if the mode will not be selected
                for output_state, prob_ampli in self._result_dict[key][state[min_r:max_r]].items():
                    nsv[state.set_slice(slice(min_r, max_r), output_state)] += prob_ampli * sv[state]
            else:
                nsv[state] = sv[state]
        return nsv

    def probs(self, input_state) -> BSDistribution:
        return _to_bsd(self.evolve(input_state))

    def probs_svd(self, svd: SVDistribution, progress_callback: Callable) -> Dict:
        res_bsd = BSDistribution()
        for sv, p_sv in svd.items():
            res = self.probs(sv)
            for bs, p_res in res.items():
                res_bsd[bs] += p_res*p_sv
        return {"results": res_bsd}

    def evolve(self, input_state) -> StateVector:
        self.compile(input_state)
        assert self._out.m == input_state.m, "Loss channels cannot be used with state amplitude"
        return self._out

    def compile(self, input_states: Union[BasicState, StateVector]) -> bool:
        if isinstance(input_states, BasicState):
            sv = StateVector(input_states)
        else:
            sv = input_states
        var = [float(p) for _, c in self._C for p in c.get_parameters()]
        if self._compiled_input == (var, sv):
            return False
        self._compiled_input = copy.copy((var, sv))
        for r, c in self._C:
            if hasattr(c, "apply"):
                sv = c.apply(r, sv)
            else:
                sv = self.apply(sv, r, c)
        self._out = sv
        return True
