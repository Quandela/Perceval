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

from .abstract_processor import AProcessor, ProcessorType
from .source import Source
from .circuit import ACircuit
from perceval.utils import SVDistribution, BasicState, StateVector, global_params, Parameter
from perceval.backends import Backend, BACKEND_LIST
from typing import Dict, Callable, Type, List


class Processor(AProcessor):
    """Generic definition of processor as sources + circuit + postprocess
    """

    def __init__(self, backend_name: str, circuit: ACircuit, source: Source = Source(), post_select_fn: Callable = None,
                 heralds: Dict[int, int] = {}):
        r"""Define a processor with sources connected to the circuit and possible post_selection

        :param sources: a list of Source used by the processor
        :param circuit: a circuit define the processor internal logic
        :param post_select_fn: a post-selection function
        """
        super().__init__()
        self._source = source
        self._circuit = circuit
        # Mode post selection: expect at least # modes with photons in output
        self._min_mode_post_select = None
        # Logical post selection
        self._post_select = post_select_fn
        self._heralds = heralds
        self._inputs_map: SVDistribution = None
        self._in_port_names = {}
        self._out_port_names = {}
        self._simulator = None
        self.set_backend(backend_name)

    def set_backend(self, backend_name):
        assert backend_name in BACKEND_LIST, f"Simulation backend '{backend_name}' does not exist"
        self._simulator = BACKEND_LIST[backend_name](self._circuit)

    def type(self) -> ProcessorType:
        return ProcessorType.SIMULATOR

    @property
    def is_remote(self) -> bool:
        return False

    def mode_post_selection(self, n: int):
        super().mode_post_selection(n)
        self._min_mode_post_select = n

    @property
    def m(self) -> int:
        return self._circuit.m - len(self._heralds)

    @property
    def post_select_fn(self):
        return self._post_select

    def with_input(self, input_state: BasicState) -> None:
        self._inputs_map = None
        expected_input_length = self.m
        assert len(input_state) == expected_input_length, \
            f"Input length not compatible with circuit (expects {expected_input_length}, got {len(input_state)})"
        input_idx = 0
        expected_photons = 0
        for k in range(self._circuit.m):
            distribution = SVDistribution(StateVector("|0>"))
            if k in self._heralds:
                if self._heralds[k] == 1:
                    distribution = self._source.probability_distribution()
                    expected_photons += 1
            else:
                if input_state[input_idx] > 0:
                    distribution = self._source.probability_distribution()
                    expected_photons += 1
                input_idx += 1
            # combine distributions
            if self._inputs_map is None:
                self._inputs_map = distribution
            else:
                self._inputs_map *= distribution

        self._min_mode_post_select = expected_photons
        if 'mode_post_select' in self._parameters:
            self._min_mode_post_select = self._parameters['mode_post_select']

    def set_port_names(self, in_port_names: Dict[int, str], out_port_names: Dict[int, str] = {}):
        self._in_port_names = in_port_names
        self._out_port_names = out_port_names

    @property
    def source_distribution(self):
        return self._inputs_map

    @property
    def circuit(self):
        return self._circuit

    @property
    def source(self):
        return self._source

    def filter_herald(self, s: BasicState, keep_herald: bool = False) -> BasicState:
        if not self._heralds or keep_herald:
            return s
        new_state = []
        for idx, k in enumerate(s):
            if idx not in self._heralds:
                new_state.append(k)
        return BasicState(new_state)

    def _run_checks(self, command_name: str):
        assert self._simulator is not None, "Simulator is missing"
        assert self._inputs_map is not None, "Input is missing, please call with_inputs()"
        assert self.available_sampling_method == command_name, \
            f"Cannot call {command_name}(). Available method is {self.available_sampling_method} "

    def samples(self, count: int, progress_callback=None) -> Dict:
        self._run_checks("samples")
        output = []
        not_selected_physical = 0
        not_selected = 0
        selected_inputs = self._inputs_map.sample(count, non_null=False)
        idx = 0
        while len(output) < count:
            selected_input = selected_inputs[idx][0]
            idx += 1
            if idx == count:
                idx = 0
                selected_inputs = self._inputs_map.sample(count, non_null=False)
            if not self._state_preselected_physical(selected_input):
                not_selected_physical += 1
                continue
            sampled_state = self._simulator.sample(selected_input)
            if not self._state_selected_physical(sampled_state):
                not_selected_physical += 1
                continue
            if self._state_selected(sampled_state):
                output.append(self.filter_herald(sampled_state))
            else:
                not_selected += 1
            if progress_callback:
                progress_callback(len(output)/count, "sampling")
        physical_perf = (count + not_selected) / (count + not_selected + not_selected_physical)
        logical_perf = count / (count + not_selected)
        return {'results': output, 'physical_perf': physical_perf, 'logical_perf': logical_perf}

    def probs(self, progress_callback: Callable = None) -> Dict:
        self._run_checks("probs")
        output = SVDistribution()
        idx = 0
        input_length = len(self._inputs_map)
        physical_perf = 1
        p_logic_discard = 0

        for input_state, input_prob in self._inputs_map.items():
            for (output_state, p) in self._simulator.allstateprob_iterator(input_state):
                if p < global_params['min_p']:
                    continue
                output_prob = p * input_prob
                if not self._state_selected_physical(output_state):
                    physical_perf -= output_prob
                    continue
                if self._state_selected(output_state):
                    output[self.filter_herald(output_state)] += output_prob
                else:
                    p_logic_discard += output_prob
            idx += 1
            if progress_callback:
                progress_callback(idx/input_length)
        if physical_perf < global_params['min_p']:
            physical_perf = 0
        all_p = sum(v for v in output.values())
        if all_p == 0:
            return {'results': output, 'physical_perf': physical_perf}
        logical_perf = 1 - p_logic_discard / (p_logic_discard + all_p)
        # normalize probabilities
        for k in output.keys():
            output[k] /= all_p
        return {'results': output, 'physical_perf': physical_perf, 'logical_perf': logical_perf}

    def _state_preselected_physical(self, input_state: BasicState):
        return input_state.n >= self._min_mode_post_select

    def _state_selected_physical(self, output_state: BasicState) -> bool:
        modes_with_photons = len([n for n in output_state if n > 0])
        return modes_with_photons >= self._min_mode_post_select

    def _state_selected(self, state: BasicState) -> bool:
        """
        Computes if the state is selected given heralds and post selection function
        """
        for m, v in self._heralds.items():
            if state[m] != v:
                return False
        if self._post_select is not None:
            return self._post_select(state)
        return True

    @property
    def available_sampling_method(self) -> str:
        preferred_command = self._simulator.preferred_command()
        if preferred_command == 'samples':
            return 'samples'
        return 'probs'

    def get_circuit_parameters(self) -> Dict[str, Parameter]:
        return {p.name: p for p in self._circuit.get_parameters()}

    def set_circuit_parameters(self, params: Dict[str, float]) -> None:
        circuit_params = self.get_circuit_parameters()
        for param_name, param_value in params.items():
            if param_name in circuit_params:
                circuit_params[param_name].set_value(param_value)
