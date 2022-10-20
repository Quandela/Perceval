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

from perceval.utils import StateVector, BasicState
from perceval.components import ACircuit
from perceval.backends import BACKEND_LIST


class StepperBackend:
    """Step-by-step circuit propagation algorithm, main usage is on a circuit, but could work in degraded mode
       on a circuit defined with a unitary matrix.
       - Use Naive backend for actual calculation of each component for non-symbolic resolution.
       - TODO: Use SLOS backend for symbolic computation
    """

    def __init__(self,
                 cu: list,
                 backend_name="Naive"):
        self._out = None
        self._C = cu
        self.backend = BACKEND_LIST[backend_name]
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
            sim_c = self.backend(c.compute_unitary(use_symbolic=False))
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
        assert self._out.m == input_state.m, "Loss channels cannot be used with state amplitude"
        return self._out

    def prob(self,
             input_state: Union[BasicState, StateVector],
             output_state: BasicState,
             n: int = None,
             skip_compile: bool = False) -> float:
        if not skip_compile:
            self.compile(input_state)
        if self._out.m == input_state.m:
            return self.prob_be(input_state, output_state, n)
        # Else we need a state reduction
        return sum(self.prob_be(input_state, state) for state in self._out if state[:output_state.m] == output_state)

    def probampli(self,
                  input_state: Union[BasicState, StateVector],
                  output_state: BasicState,
                  n: int = None) -> complex:
        self.compile(input_state)
        assert self._out.m == input_state.m, "Loss channels cannot be used with state amplitude"
        return self.probampli_be(input_state, output_state, n)

    def allstateprob_iterator(self, input_state):
        self.compile(input_state)
        n = max(input_state.n) if isinstance(input_state, StateVector) else input_state.n
        out = StateVector()
        out.update({BasicState([i]): 1 for i in range(n + 1)})  # Give it all useful n
        for output_state in self.allstate_iterator(out):
            yield output_state, self.prob(input_state, output_state, skip_compile=True)

    @staticmethod
    def preferred_command() -> str:
        return 'evolve'
