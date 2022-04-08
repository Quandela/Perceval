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
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from typing import List, Union
import copy

from .template import Backend
from perceval.utils import StateVector, BasicState, Matrix
from perceval.components import ACircuit
from .naive import NaiveBackend

import quandelibc as qc


class StepperBackend(Backend):
    """Step-by-step circuit propagation algorithm, main usage is on a circuit, but could work in degraded mode
       on a circuit defined with a unitary matrix.
       - Use Naive backend for actual calculation of each component for non-symbolic resolution.
       - TODO: Use SLOS backend for symbolic computation
    """

    def __init__(self,
                 cu: Union[ACircuit, Matrix],
                 use_symbolic: bool = None,
                 n: int = None,
                 mask: list = None):
        self._out = None
        super().__init__(cu, use_symbolic, n, mask)

    name = "Stepper"
    supports_symbolic = False
    supports_circuit_computing = True

    def apply(self, sv: StateVector, r: List[int], c: ACircuit) -> StateVector:
        """Apply a circuit on a StateVector generating another StateVector

        :param sv: input StateVector
        :param r: range of port for the circuit corresponding to StateVector position
        :param c: a circuit
        :return: evolved StateVector
        """
        min_r = r[0]
        max_r = r[-1]
        # build list of fockstates corresponding to subspace [min_r:max_r]
        sub_input_state = set()
        for state in sv:
            sub_input_state.add(BasicState(state[min_r:max_r]))
        # get circuit probability for these input_states
        sim_c = NaiveBackend(c.U, use_symbolic=self._use_symbolic)
        mapping_input_output = {}
        for input_state in sub_input_state:
            mapping_input_output[input_state] = {}
            for output_state in sim_c.allstate_iterator(input_state):
                mapping_input_output[input_state][output_state] = sim_c.probampli(input_state, output_state)
        # now rebuild the new state vector
        nsv = StateVector()
        for state in sv:
            input_state = state[min_r:max_r]
            for output_state, prob_ampli in mapping_input_output[input_state].items():
                nsv[BasicState(state.set_slice(slice(min_r, max_r), output_state))] += prob_ampli*sv[state]
        return nsv

    def compile(self, input_states: Union[BasicState, StateVector]) -> bool:
        if isinstance(input_states, BasicState):
            sv = StateVector(input_states)
        else:
            sv = input_states
        var = [float(p) for p in self._C.get_parameters()]
        if self._compiled_input == (var, sv):
            return False
        self._compiled_input = copy.copy((var, sv))
        for r, c in self._C:
            if not c.delay_circuit:
                # nsv = sv.align(r)
                sv = self.apply(sv, r, c)
            else:
                sv.apply_delta_t(r[0], float(c._dt))
        self._out = sv
        return True

    def prob_be(self, input_state, output_state, n=None, output_idx=None):
        return abs(self.probampli_be(input_state, output_state, n, output_idx))**2

    def probampli_be(self, _, output_state, n=None, output_idx=None):
        if output_state not in self._out:
            return 0
        return self._out[output_state]
