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

from .abstract_algorithm import AAlgorithm
from .sampler import Sampler
from perceval.utils import BasicState, allstate_iterator
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
        elif isinstance(output_states, list):
            self.output_states_list = output_states
        elif isinstance(output_states, dict):
            self.output_states_list = []
            for k, v in output_states.items():
                self._mapping[k] = v
                self.output_states_list.append(k)
        elif output_states == '*':
            out_set = set()
            for input_state in self.input_states_list:
                for os in allstate_iterator(input_state):
                    out_set.add(os)
            self.output_states_list = list(out_set)  # All states will be used in compute()
        # Setup output state selection on clicks
        if output_states == '*':
            min_output_photon_count = 1
        else:
            min_output_photon_count = processor.m
            for ostate in self.output_states_list:
                modes_with_photons = len([n for n in ostate if n > 0])
                min_output_photon_count = min(modes_with_photons, min_output_photon_count)
        processor.mode_post_selection(min_output_photon_count)

    def compute(self, normalize=False, expected=None, progress_callback=None):
        """
        Iterate through the input states, generate (post-selected) output states and calculate distance with expected (if
        provided)
        """
        probs_res = {}
        logical_perf = []
        if expected is not None:
            normalize = True
            self.error_rate = 0

        # Compute probabilities for all input states
        for idx, i_state in enumerate(self.input_states_list):
            self._processor.with_input(i_state)
            probs_output = self._sampler.probs()
            probs = probs_output['results']
            probs_res[i_state] = probs
            if 'logical_perf' in probs_output:
                logical_perf.append(probs_output['logical_perf'])
            else:
                logical_perf.append(1)
            if progress_callback is not None:
                progress_callback((idx+1)/len(self.input_states_list))

        # Create a distribution matrix and compute performance / error rate if needed
        self._distribution = np.zeros((len(self.input_states_list), len(self.output_states_list)))
        for iidx, i_state in enumerate(self.input_states_list):
            sum_p = 0
            for oidx, o_state in enumerate(self.output_states_list):
                if o_state in probs_res[i_state]:
                    self._distribution[iidx, oidx] = probs_res[i_state][o_state]
                    sum_p += probs_res[i_state][o_state]
            if expected is not None:
                if i_state in expected:
                    expected_o = expected[i_state]
                elif i_state in self._mapping and self._mapping[i_state] in expected:
                    expected_o = expected[self._mapping[i_state]]
                if not isinstance(expected_o, BasicState):
                    for k, v in self._mapping.items():
                        if v == expected_o:
                            expected_o = k
                            break
                if sum_p > 0:
                    self.error_rate += 1 - self._distribution[iidx, self.output_states_list.index(expected_o)]/sum_p
            if normalize and sum_p != 0:
                self._distribution[iidx, :] /= sum_p
        self.performance = min(logical_perf)
        output = {'results': self._distribution, 'input_states': self.input_states_list,
                  'output_states': self.output_states_list, 'performance': self.performance}
        if expected is not None:
            self.error_rate /= len(self.input_states_list)
            output['error_rate'] = self.error_rate
        return output

    @property
    def distribution(self):
        if self._distribution is None:
            self.compute()
        return self._distribution

    def col(self, output_state: BasicState) -> int:
        if output_state in self.output_states_list:
            return self.output_states_list.index(output_state)
        return None
