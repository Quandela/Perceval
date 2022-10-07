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

import copy
from typing import List

import numpy as np

from .abstract_algorithm import AAlgorithm
from .sampler import Sampler
from perceval.utils import BasicState, StateVector
from perceval.components import AProcessor


class Analyzer(AAlgorithm):
    def __init__(self, processor: AProcessor, input_states, output_states=None, mapping=None):
        """
            Initialization of a processor analyzer
            `simulator` is a simulator instance initialized for the circuit
            `input_states` is a list of BasicStates or a mapping BasicState => name
            `output_states` is a list of BasicStates or a mapping BasicState => name, if missing input is taken,
                            if "*" all possible target states are generated
            `mapping` is a mapping of FockState => name for display
            `post_select_fn` is a post selection function
        """
        if mapping is None:
            mapping = {}
        processor.mode_post_selection(0)
        super().__init__(processor)
        self._sampler = Sampler(processor)
        self._mapping = mapping
        self.performance = None
        self.error_rate = None
        self._distribution = None

        # Enrich mapping and create self.input_state_list
        if isinstance(input_states, dict):
            self.input_states_list = []
            for k, v in input_states.items():
                self._mapping[k] = v
                self.input_states_list.append(k)
        elif isinstance(input_states, list):
            self.input_states_list = input_states
        else:
            raise TypeError("input_states must be a list or a dictionary")

        # Test input_states_list
        for input_state in self.input_states_list:
            assert isinstance(input_state, BasicState), "input_states should contain BasicStates"
            assert input_state.m == self._processor.m, "Incorrect BasicState size"

        if output_states is None:
            self.output_states_list = self.input_states_list
        elif isinstance(output_states, dict):
            self.output_states_list = []
            for k, v in output_states.items():
                self._mapping[k] = v
                self.output_states_list.append(k)
        elif output_states == '*':
            self.output_states_list = None  # All states will be used in compute()

    def _generate_displayable_output_states(self, expected) -> List:
        """
        Generate a list of output states to consider for display.
        Merges the output states from self.output_states_list and expected if they exist
        When none is defined, returns an empty list
        """
        disp_out_s = set()
        if self.output_states_list is not None:
            for s in self.output_states_list:
                disp_out_s.add(s)
        if expected is not None:
            for s in expected.values():
                disp_out_s.add(s)
        return list(disp_out_s)

    def compute(self, normalize=False, expected=None, progress_callback=None):
        """
        Iterate through the input states, generate (post-selected) output states and calculate distance with expected (if
        provided)
        """
        probs_res = {}
        perf_res = []
        expected_out_states_to_display = self._generate_displayable_output_states(expected)
        out_states_to_display = set()
        if expected is not None:
            normalize = True

        for idx, i_state in enumerate(self.input_states_list):
            self._processor.with_input(i_state)
            probs_output = self._sampler.probs()
            probs = probs_output['results']
            probs_res[i_state] = probs
            for s in probs:
                if len(expected_out_states_to_display) == 0 or s[0] in expected_out_states_to_display:
                    out_states_to_display.add(s[0])
            if progress_callback is not None:
                progress_callback((idx+1)/len(self.input_states_list))
        self.output_states_list = list(out_states_to_display)

        # Create distribution matrix
        self._distribution = np.zeros((len(self.input_states_list), len(self.output_states_list)))
        for iidx, i_state in enumerate(self.input_states_list):
            sum_p = 0
            for oidx, o_state in enumerate(self.output_states_list):
                if StateVector(o_state) in probs_res[i_state]:
                    self._distribution[iidx, oidx] = probs_res[i_state][o_state]
                    sum_p += probs_res[i_state][o_state]
            if normalize:
                self._distribution[iidx, :] /= sum_p
            perf_res.append(sum_p)
        self.performance = min(perf_res)

    # def compute_(self, normalize=False, expected=None, progress_callback=None):
    #     """
    #         Go through the input states, generate (post-selected) output states and calculate if provided
    #         distance with expected
    #     """
    #     self._distribution = np.zeros((len(self.input_states_list), len(self.output_states_list)))
    #     computation_count = 0
    #     total_count = len(self.input_states_list) * len(self.output_states_list)
    #     if expected is not None:
    #         self._expected_distribution = np.zeros((len(self.input_states_list), len(self.output_states_list)))
    #         self.performance = 1
    #         self.error_rate = 0
    #     for iidx, istate in enumerate(self.input_states_list):
    #         sump = 1e-6
    #         if expected is not None:
    #             if istate in expected:
    #                 expected_o = expected[istate]
    #             elif istate in self._mapping and self._mapping[istate] in expected:
    #                 expected_o = expected[self._mapping[istate]]
    #             if not isinstance(expected_o, BasicState):
    #                 for k, v in self._mapping.items():
    #                     if v == expected_o:
    #                         expected_o = k
    #                         break
    #             self._expected_distribution[iidx, self.output_states_list.index(expected_o)] = 1
    #         for oidx, ostate in enumerate(self.output_states_list):
    #             if self._post_select_fn is None or self._post_select_fn(ostate):
    #                 if istate.n == ostate.n:
    #                     self._distribution[iidx, oidx] = self.prob(istate, ostate)  # job run synchronously
    #                     if expected is not None and self._expected_distribution[iidx, oidx]:
    #                         found_in_row = self._distribution[iidx, oidx]
    #                         if self._distribution[iidx, oidx] < self.performance:
    #                             self.performance = self._distribution[iidx, oidx]
    #                 sump += self._distribution[iidx, oidx]
    #             computation_count += 1
    #             if progress_callback:
    #                 progress_callback(computation_count/total_count)
    #         if normalize or expected is not None:
    #             self._distribution[iidx, :] /= sump
    #         if expected is not None:
    #             self.error_rate += 1-found_in_row/sump
    #     if expected is not None:
    #         self.error_rate /= len(self.input_states_list)
    #     return self

    @property
    def distribution(self):
        if self._distribution is None:
            self.compute()
        return self._distribution
