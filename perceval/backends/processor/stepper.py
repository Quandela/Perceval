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

import quandelibc as qc


class StepperBackend:
    """
    :param cu: A unitary circuit or a list of components.
    :param m: The size of the circuit. Needed if a list is given.
    :param backend_name: The name of the backend that will be used for step by step computation.
    :param mode_post_selection: The minimal number of modes that will be needed to keep a state.
     Basically stops computing for states not having enough photons, but do not remove them.

    Step-by-step circuit propagation algorithm, main usage is on a circuit, but could work in degraded mode
    on a list of components [(r, comp)].
    """

    def __init__(self,
                 cu: Union[list, ACircuit],
                 m: int = None,
                 backend_name="Naive",
                 mode_post_selection=0):
        self._out = None
        self._C = cu
        self._backend = BACKEND_LIST[backend_name]
        self._result_dict = {c.describe(): {'_set': set()} for r, c in self._C}
        self._compiled_input = None
        self.mode_post_selection = mode_post_selection
        if isinstance(cu, ACircuit):
            self.m = cu.m
        else:
            assert m is not None, "Please specify the number of modes of the circuit"
            self.m = m

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
                           for sliced_state in (state[min_r:max_r],)
                           if sliced_state not in self._result_dict[key]['_set']
                           and state[:self.m].n >= self.mode_post_selection}
        # get circuit probability for these input_states
        if sub_input_state:
            sim_c = self._backend(c.compute_unitary(use_symbolic=False))
            mapping_input_output = {input_state:
                                    {output_state: sim_c.probampli(input_state, output_state)
                                        for output_state in sim_c.allstate_iterator(input_state)}
                                    for input_state in sub_input_state}
            self._result_dict[key].update(mapping_input_output)  # Union of the dictionaries
            self._result_dict[key]['_set'] |= sub_input_state  # Union of sets
        # now rebuild the new state vector
        nsv = StateVector()
        # May be faster in c++ (impossible to use comprehension here due to successive additions)
        for state in sv:
            if state[:self.m].n >= self.mode_post_selection:  # Useless to compute if the mode will not be selected
                for output_state, prob_ampli in self._result_dict[key][state[min_r:max_r]].items():
                    nsv[state.set_slice(slice(min_r, max_r), output_state)] += prob_ampli * sv[state]
            else:
                nsv[state] = sv[state]
        return nsv

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

    def prob_be(self, input_state, output_state):
        return abs(self.probampli_be(input_state, output_state))**2

    def probampli_be(self, _, output_state):
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
             skip_compile: bool = False,
             progress_callback=None) -> float:
        if not skip_compile:
            self.compile(input_state)
        if self._out.m == input_state.m:
            return self.prob_be(input_state, output_state)
        # Else we need a state reduction
        return sum(self.prob_be(input_state, state) for state in self._out if state[:output_state.m] == output_state)

    def probampli(self,
                  input_state: Union[BasicState, StateVector],
                  output_state: BasicState) -> complex:
        self.compile(input_state)
        assert self._out.m == input_state.m, "Loss channels cannot be used with state amplitude"
        return self.probampli_be(input_state, output_state)

    def allstateprob_iterator(self, input_state):
        self.compile(input_state)
        n = max(input_state.n) if isinstance(input_state, StateVector) else input_state.n
        out = StateVector()
        out.update({BasicState([i]): 1 for i in range(n + 1)})  # Give it all useful n
        for output_state in self.allstate_iterator(out):
            yield output_state, self.prob(input_state, output_state, skip_compile=True)

    def allstate_iterator(self, input_state: Union[BasicState, StateVector]) -> BasicState:
        """Iterator on all possible output states compatible with mask generating StateVector

        :param input_state: a given input state vector
        :return: list of output_state
        """
        m = self.m
        ns = input_state.n
        if not isinstance(ns, list):
            ns = [ns]
        for n in ns:
            output_array = qc.FSArray(m, n)
            for output_idx, output_state in enumerate(output_array):
                yield BasicState(output_state)

    @staticmethod
    def preferred_command() -> str:
        return 'evolve'

    @staticmethod
    def available_commands() -> List[str]:
        return ['prob', 'prob_be', 'probampli', 'probampli_be', 'evolve']
