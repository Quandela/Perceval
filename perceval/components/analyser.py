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

import numpy as np
from tabulate import tabulate
import copy

from perceval.utils.format import simple_float
from perceval.utils.statevector import BasicState


class CircuitAnalyser:
    def __init__(self, simulator, input_states, output_states=None, mapping=None, post_select_fn=None):
        """
            Initialization of Circuit Analyzer
            `simulator` is a simulator instance initialized for the circuit
            `input_states` is a list of FockStates or a mapping fockstate => name
            `output_states` is a list of FockState or a mapping fockstate => name, if missing input is taken,
                            if "*" all possible target states are generated
            `mapping` is a mapping of FockState => name for display
            `post_select_fn` is a post selection function
        """
        self._simulator = simulator
        if mapping is None:
            self._mapping = {}
        else:
            self._mapping = mapping
        if isinstance(input_states, dict):
            self.input_states_list = []
            for k, v in input_states.items():
                self._mapping[k] = v
                self.input_states_list.append(k)
        else:
            self.input_states_list = input_states
        for input_state in self.input_states_list:
            assert isinstance(input_state, BasicState), "input_states should be BasicStates"
            assert input_state.m == simulator.m, "incorrect BasicState"
        if output_states is None:
            self.output_states_list = self.input_states_list
        elif output_states == "*":
            outs = set()
            self.output_states_list = []
            explored_n = set()
            for input_state in self.input_states_list:
                if input_state.n in explored_n:
                    continue
                explored_n.add(input_state.n)
                for output_state in simulator.allstate_iterator(input_state):
                    if post_select_fn is None or post_select_fn(output_state):
                        if output_state not in outs:
                            outs.add(output_state)
                            self.output_states_list.append(copy.copy(output_state))
        elif isinstance(output_states, dict):
            self.output_states_list = []
            for k, v in output_states.items():
                self._mapping[k] = v
                self.output_states_list.append(k)
        else:
            self.output_states_list = output_states
        self._post_select_fn = post_select_fn
        self.performance = None
        self.error_rate = None
        self._distribution = None

    def compute(self, normalize=False, expected=None):
        """
            Go through the input states, generate (post-selected) output states and calculate if provided
            distance with expected
        """
        self._distribution = np.zeros((len(self.input_states_list), len(self.output_states_list)))
        if expected is not None:
            self._expected_distribution = np.zeros((len(self.input_states_list), len(self.output_states_list)))
            self.performance = 1
            self.error_rate = 0
        for iidx, istate in enumerate(self.input_states_list):
            sump = 1e-6
            if expected is not None:
                if istate in expected:
                    expected_o = expected[istate]
                elif istate in self._mapping and self._mapping[istate] in expected:
                    expected_o = expected[self._mapping[istate]]
                if not isinstance(expected_o, BasicState):
                    for k, v in self._mapping.items():
                        if v == expected_o:
                            expected_o = k
                            break
                self._expected_distribution[iidx, self.output_states_list.index(expected_o)] = 1
            for oidx, ostate in enumerate(self.output_states_list):
                if self._post_select_fn is None or self._post_select_fn(ostate):
                    if istate.n == ostate.n:
                        self._distribution[iidx, oidx] = self._simulator.prob(istate, ostate)
                        if expected is not None and self._expected_distribution[iidx, oidx]:
                            found_in_row = self._distribution[iidx, oidx]
                            if self._distribution[iidx, oidx] < self.performance:
                                self.performance = self._distribution[iidx, oidx]
                    sump += self._distribution[iidx, oidx]
            if normalize or expected is not None:
                self._distribution[iidx, :] /= sump
            if expected is not None:
                self.error_rate += 1-found_in_row/sump
        if expected is not None:
            self.error_rate /= len(self.input_states_list)
        return self

    def pdisplay(self, output_format="text", nsimplify=True, precision=1e-6):
        distribution = self.distribution
        d = []
        for iidx, _ in enumerate(self.input_states_list):
            d.append([simple_float(f, nsimplify=nsimplify, precision=precision)[1]
                      for f in list(distribution[iidx])])
        return tabulate(d, headers=[self._mapping.get(o, str(o)) for o in self.output_states_list],
                        showindex=[self._mapping.get(i, str(i)) for i in self.input_states_list],
                        tablefmt=output_format == "text" and "pretty" or output_format)

    @property
    def distribution(self):
        if self._distribution is None:
            self.compute()
        return self._distribution
