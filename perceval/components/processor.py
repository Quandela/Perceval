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

from .source import Source
from .linear_circuit import ALinearCircuit, Circuit
from .base_components import PERM
from .port import APort, PortLocation, Herald
from perceval.utils import SVDistribution, StateVector, AnnotatedBasicState, global_params
from perceval.backends import Backend
from typing import Dict, Callable, Type, Literal, Union


class UnavailableModeException(Exception):
    def __init__(self, mode: Union[int, list[int]], reason: str = None):
        because = ''
        if reason:
            because = f' because: {reason}'
        super().__init__(f"Mode(s) {mode} not available{because}")


class Processor:
    """
        Generic definition of processor as sources + circuit
    """
    def __init__(self, source: Source = Source()):
        r"""Define a processor with sources connected to the circuit and possible post_selection

        :param source: the Source used by the processor
        """
        self._source = source
        self._components = []  # Any type of components, not only linear ones
        self._in_ports = {}
        self._out_ports = {}
        self._n_modes = 0  # number of modes

        self._anon_herald_num = 0  # This is not a herald count!

        # self._closed_photonic_modes = []

        # self._circuit = circuit
        # self._post_select = post_select_fn
        # self._heralds = heralds
        # self._inputs_map = None
        # for k in range(circuit.m):
        #     if k in sources:
        #         distribution = sources[k].probability_distribution()
        #     else:
        #         distribution = SVDistribution(StateVector("|0>"))
        #     # combine distributions
        #     if self._inputs_map is None:
        #         self._inputs_map = distribution
        #     else:
        #         self._inputs_map *= distribution
        # self._in_port_names = {}
        # self._out_port_names = {}

    def add(self, mode_mapping, component):
        """
        Add a component (linear or non linear)
        Checks if it's possible:
        - No output is set on used modes
        - Should fail if lock_inputs has been called and the component uses new modes or defines new inputs
        """
        if isinstance(mode_mapping, int):
            mode_mapping = list(range(mode_mapping, mode_mapping + component.m))
        # TODO mapping between port names and indexes
        assert isinstance(mode_mapping, list) or isinstance(mode_mapping, tuple), "mode_mapping must be a list"

        for mode in mode_mapping:
            if not self.is_mode_connectible(mode):
                raise UnavailableModeException(mode)

        perm_modes, perm_component = self._generate_permutation(mode_mapping)
        if perm_component is not None:
            self._components.append((perm_modes, perm_component))

        self._n_modes = max(self._n_modes, max(mode_mapping)+1)
        component = Circuit(component.m).add(0, component, merge=False)
        sorted_modes = list(range(min(mode_mapping), min(mode_mapping)+component.m))
        self._components.append((sorted_modes, component))
        return self

    def _generate_permutation(self, mode_mapping):
        min_m = min(mode_mapping)
        max_m = max(mode_mapping)
        missing_modes = [x for x in list(range(min_m, max_m+1)) if x not in mode_mapping]
        mode_mapping.extend(missing_modes)
        perm_modes = list(range(min_m, min_m+len(mode_mapping)))
        perm_norm = [m - min_m for m in mode_mapping]
        perm_vect = [perm_norm.index(i) for i in range(len(perm_norm))]
        if mode_mapping == perm_modes:
            return perm_modes, None  # No need for a permutation, modes are already sorted

        return perm_modes, PERM(perm_vect)


    def add_port(self, m, port: APort, location: PortLocation = PortLocation.in_out):
        port_range = list(range(m, m + port.m))
        assert port.supports_location(location), f"Port is not compatible with location '{location.name}'"
        if port.name is None and isinstance(port, Herald):
            port._name = f'herald{self._anon_herald_num}'
            self._anon_herald_num += 1

        if location == PortLocation.in_out or location == PortLocation.input:
            if not self.are_modes_free(port_range, PortLocation.input):
                raise UnavailableModeException(port_range, "Another port overlaps")
            self._in_ports[port] = port_range

        if location == PortLocation.in_out or location == PortLocation.output:
            if not self.are_modes_free(port_range, PortLocation.output):
                raise UnavailableModeException(port_range, "Another port overlaps")
            self._out_ports[port] = port_range

    @property
    def _closed_photonic_modes(self):
        output = [False] * self._n_modes
        for port, m_range in self._out_ports.items():
            if port.is_output_photonic_mode_closed():
                for i in m_range:
                    output[i] = True
        return output

    def is_mode_connectible(self, mode: int):
        if mode < 0:
            return False
        if mode >= self._n_modes:
            return True
        return not self._closed_photonic_modes[mode]

    def are_modes_free(self, mode_range, location: PortLocation = PortLocation.output):
        """
        Returns True if all modes in mode_range are free of ports, for a given location (input, output or both)
        """
        if location == PortLocation.in_out or location == PortLocation.input:
            for m in mode_range:
                if self.get_input_port(m) is not None:
                    return False
        if location == PortLocation.in_out or location == PortLocation.output:
            for m in mode_range:
                if self.get_output_port(m) is not None:
                    return False
        return True

    def get_input_port(self, mode):
        for port, mode_range in self._in_ports.items():
            if mode in mode_range:
                return port
        return None

    def get_output_port(self, mode):
        for port, mode_range in self._out_ports.items():
            if mode in mode_range:
                return port
        return None

    @property
    def source_distribution(self):
        return self._inputs_map

    @property
    def source(self):
        return self._source

    def filter_herald(self, s: AnnotatedBasicState, keep_herald: bool) -> StateVector:
        if not self._heralds or keep_herald:
            return StateVector(s)
        new_state = []
        for idx, k in enumerate(s):
            if idx not in self._heralds:
                new_state.append(k)
        return StateVector(new_state)

    def run(self, simulator_backend: Type[Backend], keep_herald: bool=False):
        """
            calculate the output probabilities - returns performance, and output_maps
        """
        # first generate all possible outputs
        sim = simulator_backend(self._circuit.compute_unitary(use_symbolic=False))
        # now generate all possible outputs
        outputs = SVDistribution()
        for input_state, input_prob in self._inputs_map.items():
            for (output_state, p) in sim.allstateprob_iterator(input_state):
                if p > global_params['min_p'] and self._state_selected(output_state):
                    outputs[self.filter_herald(output_state, keep_herald)] += p*input_prob
        all_p = sum(v for v in outputs.values())
        if all_p == 0:
            return 0, outputs
        # normalize probabilities
        for k in outputs.keys():
            outputs[k] /= all_p
        return all_p, outputs

    def _state_selected(self, state: AnnotatedBasicState) -> bool:
        """
        Computes if the state is selected given heralds and post selection function
        """
        for m, v in self._heralds.items():
            if state[m] != v:
                return False
        if self._post_select is not None:
            return self._post_select(state)
        return True
