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
from __future__ import annotations
import numpy as np

from .abstract_algorithm import AAlgorithm
from .sampler import Sampler
from perceval.utils import BasicState, allstate_iterator, Matrix
from perceval.components import AProcessor


class Analyzer(AAlgorithm):
    """
    Analyses a set of input states vs output states probabilities.

    :param processor: the processor to analyse
    :param input_states: list of BasicStates or a mapping {BasicState: name}
    :param output_states: list of output states. Valid values are:
                          * None (then, the input states are taken as output states)
                          * a list of BasicState
                          * a mapping {BasicState: name}
                          * the string "*" meaning oll possible target states are generated
    :param mapping: optional mapping {BasicState: name} used for display
    :param kwargs: as the Analyzer internally uses a Sampler instance, it needs a "max_shots_per_call" value
    """

    def __init__(self, processor: AProcessor,
                 input_states: list[BasicState] | dict[BasicState, str],
                 output_states=None,
                 mapping=None,
                 **kwargs):
        if mapping is None:
            mapping = {}
        super().__init__(processor, **kwargs)
        self._sampler = Sampler(processor, **kwargs)
        self._mapping = mapping
        self.performance = None
        self.error_rate = None
        self.fidelity = None
        self._distribution = None
        self.default_job_name = 'analyzer'

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
        # Setup output state selection on detected photons
        if output_states == '*':
            min_output_photon_count = 1  # To retrieve all non-empty states on a QPU, set filter to 1
        else:
            min_output_photon_count = processor.m
            for ostate in self.output_states_list:
                min_output_photon_count = min(ostate.n, min_output_photon_count)
        processor.min_detected_photons_filter(min_output_photon_count)

    def compute(self, normalize: bool = False, expected: dict = None, progress_callback=None):
        """
        Iterate through the input states, generate (post-selected) output states and calculate distance with expected,
        if provided.

        :param normalize: whether to normalize the output states
        :param expected: optional mapping between states in ideal case
        :param progress_callback: optional callback to inform the user of the task progress
        """
        probs_res = {}
        logical_perf = []
        has_an_empty_PD = False
        if expected is not None:
            normalize = True
            self.error_rate = 0

        # Compute probabilities for all input states
        for idx, i_state in enumerate(self.input_states_list):
            self._processor.with_input(i_state)
            job = self._sampler.probs
            job.name = f'{self.default_job_name} {idx+1}/{len(self.input_states_list)}'
            probs_output = job.execute_sync()
            probs = probs_output['results']
            if len(probs) == 0:
                has_an_empty_PD = True
            probs_res[i_state] = probs
            if 'logical_perf' in probs_output:
                logical_perf.append(probs_output['logical_perf'])
            else:
                logical_perf.append(probs_output['global_perf'])
            if progress_callback is not None:
                progress_callback((idx+1)/len(self.input_states_list))

        # Create a distribution matrix and compute performance / error rate if needed
        self._distribution = Matrix(np.zeros((len(self.input_states_list), len(self.output_states_list))))
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
                    self.error_rate += 1 - \
                        (self._distribution[iidx, self.output_states_list.index(expected_o)]/sum_p).real
            if normalize and sum_p != 0:
                self._distribution[iidx, :] /= sum_p
        self.performance = min(logical_perf)
        output = {'results': self._distribution, 'input_states': self.input_states_list,
                  'output_states': self.output_states_list, 'performance': self.performance}

        if has_an_empty_PD:
            output['performance'] = 0
        if expected is not None:
            if has_an_empty_PD:
                output['error_rate'] = None
                output['fidelity'] = None
            else:
                self.error_rate /= len(self.input_states_list)
                output['error_rate'] = self.error_rate
                self.fidelity = 1 - self.error_rate
                output['fidelity'] = self.fidelity
        return output

    @property
    def distribution(self) -> Matrix:
        """Return the truth table of the analysis. Computes it if wasn't performed beforehand.

        :return: a matrix containing the probabilities for each input vs output states
        """
        if self._distribution is None:
            self.compute()
        return self._distribution

    def col(self, output_state: BasicState) -> int | None:
        """
        Return the column number for a given output state in the distribution matrix

        :param output_state: any computed output state
        :return: the column number, or None if the output state is unknown
        """
        if output_state in self.output_states_list:
            return self.output_states_list.index(output_state)
        return None
