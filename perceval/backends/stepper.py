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
        self.result_dict = {c.describe(): {'set': set()} for r, c in self._C}

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
        max_r = r[-1] + 1
        key = c.describe()  # Can't use c; two identical pieces aren't considered equal if they aren't at the same place
        # build list of never visited fockstates corresponding to subspace [min_r:max_r]
        sub_input_state = {sliced_state for state in sv
                           for sliced_state in (BasicState(state[min_r:max_r]),)
                           if sliced_state not in self.result_dict[key]['set']}
        # get circuit probability for these input_states
        if sub_input_state != set():
            sim_c = NaiveBackend(c.compute_unitary(use_symbolic=False))
            mapping_input_output = {input_state:
                                    {output_state: sim_c.probampli(input_state, output_state)
                                        for output_state in sim_c.allstate_iterator(input_state)}
                                    for input_state in sub_input_state}
            self.result_dict[key] |= mapping_input_output  # Union of the dictionaries
            self.result_dict[key]['set'] |= sub_input_state  # Union of sets
        # now rebuild the new state vector
        nsv = StateVector()
        # May be faster in c++ (impossible to use comprehension here due to successive additions)
        for state in sv:
            for output_state, prob_ampli in self.result_dict[key][state[min_r:max_r]].items():
                nsv[BasicState(state.set_slice(slice(min_r, max_r), output_state))] += prob_ampli * sv[state]
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
            if hasattr(c, "apply"):
                sv = c.apply(r, sv)
            else:
                sv = self.apply(sv, r, c)
        self._out = sv
        return True

    def prob_be(self, input_state, output_state, n=None, output_idx=None):
        return abs(self.probampli_be(input_state, output_state, n, output_idx))**2

    def probampli_be(self, _, output_state, n=None, output_idx=None):
        if output_state not in self._out:
            return 0
        return self._out[output_state]

    def evolve(self, input_state: Union[BasicState, StateVector]) -> StateVector:
        self.compile(input_state)
        return self._out

    def prob(self,
             input_state: Union[BasicState, StateVector],
             output_state: BasicState,
             n: int = None,
             skip_compile: bool = False) -> float:
        if not skip_compile:
            self.compile(input_state)
        return self.prob_be(input_state, output_state, n)

    def probampli(self,
                  input_state: Union[BasicState, StateVector],
                  output_state: BasicState,
                  n: int = None) -> complex:
        self.compile(input_state)
        return self.probampli_be(input_state, output_state, n)

    def allstateprob_iterator(self, input_state):
        skip_compile = False
        for output_state in self.allstate_iterator(input_state):
            yield output_state, self.prob(input_state, output_state, skip_compile=skip_compile)
            skip_compile = True
