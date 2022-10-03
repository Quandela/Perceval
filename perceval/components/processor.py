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

    def with_input(self, input_state: BasicState) -> None:
        self._inputs_map = None
        expected_input_length = self._circuit.m - len(self._heralds)
        assert len(input_state) == expected_input_length, \
            f"Input length not compatible with circuit (expects {expected_input_length}, got {len(input_state)})"
        input_idx = 0
        for k in range(self._circuit.m):
            distribution = SVDistribution(StateVector("|0>"))
            if k in self._heralds:
                if self._heralds[k] == 1:
                    distribution = self._source.probability_distribution()
            else:
                if input_state[input_idx] > 0:
                    distribution = self._source.probability_distribution()
                input_idx += 1
            # combine distributions
            if self._inputs_map is None:
                self._inputs_map = distribution
            else:
                self._inputs_map *= distribution

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

    def samples(self, count: int, progress_callback=None) -> List[BasicState]:
        self._run_checks("samples")
        output = []
        while len(output) < count:
            selected_input = self._inputs_map.sample()[0]
            sampled_state = self._simulator.sample(selected_input)
            if self._state_selected(sampled_state):
                output.append(self.filter_herald(sampled_state))
            if progress_callback:
                progress_callback(len(output)/count, "sampling")
        return output

    def probs(self) -> SVDistribution:
        self._run_checks("probs")
        output = SVDistribution()
        for input_state, input_prob in self._inputs_map.items():
            for (output_state, p) in self._simulator.allstateprob_iterator(input_state):
                if p > global_params['min_p'] and self._state_selected(output_state):
                    output[self.filter_herald(output_state)] += p * input_prob
        all_p = sum(v for v in output.values())
        if all_p == 0:
            return output
        # normalize probabilities
        for k in output.keys():
            output[k] /= all_p
        return output

    # def run(self, simulator_backend: Type[Backend], keep_herald: bool = False):
    #     """
    #         calculate the output probabilities - returns performance, and output_maps
    #     """
    #     # first generate all possible outputs
    #     sim = simulator_backend(self._circuit.compute_unitary(use_symbolic=False))
    #     # now generate all possible outputs
    #     outputs = SVDistribution()
    #     for input_state, input_prob in self._inputs_map.items():
    #         for (output_state, p) in sim.allstateprob_iterator(input_state):
    #             if p > global_params['min_p'] and self._state_selected(output_state):
    #                 outputs[StateVector(self.filter_herald(output_state, keep_herald))] += p * input_prob
    #     all_p = sum(v for v in outputs.values())
    #     if all_p == 0:
    #         return 0, outputs
    #     # normalize probabilities
    #     for k in outputs.keys():
    #         outputs[k] /= all_p
    #     return all_p, outputs

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
