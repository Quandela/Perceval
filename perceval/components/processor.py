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
from multipledispatch import dispatch

from .abstract_component import AComponent
from .source import Source
from .linear_circuit import ALinearCircuit, Circuit
from .base_components import PERM
from .port import APort, PortLocation, Herald
from perceval.utils import SVDistribution, StateVector, AnnotatedBasicState, BasicState, global_params
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
    Generic definition of processor as a source + components (both linear and non-linear) + ports
    """
    def __init__(self, n_moi: int, source: Source = Source()):
        r"""Define a processor with sources connected to the circuit and possible post_selection

        :param n_moi: number of modes of interest (MOI)
                      A mode of interest is any non-heralded mode
        :param source: the Source used by the processor
        """
        self._source = source
        self._components = []  # Any type of components, not only linear ones
        self._in_ports = {}
        self._out_ports = {}
        self._n_moi = n_moi  # number of modes of interest (MOI)
        self._n_heralds = 0

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

    @property
    def m(self):
        return self._n_moi + self._n_heralds

    def add(self, mode_mapping, component):
        """
        Add a component (linear or non linear)
        Checks if it's possible:
        - No output is set on used modes
        - Should fail if lock_inputs has been called and the component uses new modes or defines new inputs
        """
        if isinstance(component, Processor):
            mode_mapping = self._resolve_mode_mapping(mode_mapping, component.m - len(component.heralds))
            self._compose_processor(mode_mapping, component)
        elif isinstance(component, AComponent):
            mode_mapping = self._resolve_mode_mapping(mode_mapping, component.m)
            self._add_component(mode_mapping, component)
        else:
            raise RuntimeError(f"Cannot add {type(component)} object to a Processor")
        return self

    def _resolve_mode_mapping(self, mapping, n):
        if isinstance(mapping, int):
            mapping = list(range(mapping, mapping + n))
        assert isinstance(mapping, list) or isinstance(mapping, tuple), "mode_mapping must be a list"
        port_names = self.out_port_names
        result = []
        for value in mapping:
            if isinstance(value, int):
                result.append(value)
            if isinstance(value, str):  # Mapping between port name and index
                count = port_names.count(value)
                if count == 0:
                    raise RuntimeError(f"Port '{value}' not found in Processor")
                pos = port_names.index(value)
                for i in range(count):
                    result.append(pos)
                    pos += 1

        for mode in result:
            if not self.is_mode_connectible(mode):
                raise UnavailableModeException(mode)
        return result

    @property
    def out_port_names(self):
        result = [''] * self.m
        for port, m_range in self._out_ports.items():
            for m in m_range:
                result[m] = port.name
        return result

    def _compose_processor(self, mode_mapping, processor):
        # Remove output ports used to connect the new processor
        for i in mode_mapping:
            port = self.get_output_port(i)
            if port is not None:
                del self._out_ports[port]

        other_herald_pos = list(processor.heralds.keys())
        new_mode_index = self.m
        mapping_with_heralds = []
        j = 0
        for i in range(processor.m):
            if i in other_herald_pos:
                mapping_with_heralds.append(new_mode_index)
                new_mode_index += 1
                self._n_heralds += 1
            else:
                mapping_with_heralds.append(mode_mapping[j])
                j += 1
        perm_modes, perm_component = self._generate_permutation(mapping_with_heralds)
        if perm_component is not None:
            self._components.append((perm_modes, perm_component))
        for pos, c in processor._components:
            pos = [x + min(mapping_with_heralds) for x in pos]
            self._components.append((pos, c))
        if perm_component is not None:
            perm_inv = perm_component.copy()
            perm_inv.inverse(h=True)
            self._components.append((perm_modes, perm_inv))

        # Retrieve ports from other processor
        for port, port_range in processor._in_ports.items():
            if isinstance(port, Herald):
                self.add_port(mapping_with_heralds[port_range[0]], port, PortLocation.input)
        for port, port_range in processor._out_ports.items():
            self.add_port(mapping_with_heralds[port_range[0]], port, PortLocation.output)

    def _add_component(self, mode_mapping, component):
        perm_modes, perm_component = self._generate_permutation(mode_mapping)
        if perm_component is not None:
            self._components.append((perm_modes, perm_component))

        if isinstance(component, ALinearCircuit):
            component = Circuit(component.m).add(0, component, merge=False)
        sorted_modes = list(range(min(mode_mapping), min(mode_mapping)+component.m))
        self._components.append((sorted_modes, component))

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

    def add_herald(self, m, expected, name=None):
        self._n_heralds += 1
        if not self.are_modes_free([m], PortLocation.in_out):
            raise UnavailableModeException(m, "Another port overlaps")
        if name is None:
            name = f'herald{self._anon_herald_num}'
            self._anon_herald_num += 1
        self._in_ports[Herald(expected, name)] = [m]
        self._out_ports[Herald(expected, name)] = [m]
        return self

    def add_port(self, m, port: APort, location: PortLocation = PortLocation.in_out):
        port_range = list(range(m, m + port.m))
        assert port.supports_location(location), f"Port is not compatible with location '{location.name}'"

        if location == PortLocation.in_out or location == PortLocation.input:
            if not self.are_modes_free(port_range, PortLocation.input):
                raise UnavailableModeException(port_range, "Another port overlaps")
            self._in_ports[port] = port_range

        if location == PortLocation.in_out or location == PortLocation.output:
            if not self.are_modes_free(port_range, PortLocation.output):
                raise UnavailableModeException(port_range, "Another port overlaps")
            self._out_ports[port] = port_range
        return self

    @property
    def _closed_photonic_modes(self):
        output = [False] * self.m
        for port, m_range in self._out_ports.items():
            if port.is_output_photonic_mode_closed():
                for i in m_range:
                    output[i] = True
        return output

    def is_mode_connectible(self, mode: int):
        if mode < 0:
            return False
        if mode >= self.m:
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
    def heralds(self):
        pos = {}
        for port, port_range in self._out_ports.items():
            if isinstance(port, Herald):
                pos[port_range[0]] = port.expected
        return pos

    @property
    def source_distribution(self):
        return self._inputs_map

    @property
    def source(self):
        return self._source

    # def filter_herald(self, s: AnnotatedBasicState, keep_herald: bool) -> StateVector:
    #     if not self._heralds or keep_herald:
    #         return StateVector(s)
    #     new_state = []
    #     for idx, k in enumerate(s):
    #         if idx not in self._heralds:
    #             new_state.append(k)
    #     return StateVector(new_state)

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
        for m, v in self.heralds:
            if state[m] != v:
                return False
        return self._post_process_func(state)
