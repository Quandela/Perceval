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
from .abstract_component import AComponent
from .source import Source
from .linear_circuit import ALinearCircuit, Circuit
from .base_components import PERM
from .port import APort, PortLocation, Herald, Encoding
from .mode_connection import ModeConnectionResolver, UnavailableModeException
from perceval.utils import SVDistribution, StateVector, AnnotatedBasicState, BasicState, global_params
from perceval.backends import Backend
from typing import Dict, Type


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
        self._post_select = None

        self._anon_herald_num = 0  # This is not a herald count!
        self._is_on = False
        self._inputs_map = None

    def turn_on(self):
        if self._is_on:
            return
        self._is_on = True

        for k in range(self.m):
            port = self.get_input_port(k)
            if port is None:
                continue
            mode_range = self._in_ports[port]
            if isinstance(port, Herald):
                if port.expected:
                    distribution = self._source.probability_distribution()
                else:
                    distribution = SVDistribution(StateVector("|0>"))
            else:
                if port.encoding == Encoding.dual_ray:
                    if k == mode_range[0]:
                        distribution = self._source.probability_distribution()
                    else:
                        distribution = SVDistribution(StateVector("|0>"))
                else:
                    raise NotImplementedError(f"Not implemented for {port.encoding.name}")

            if self._inputs_map is None:
                self._inputs_map = distribution
            else:
                self._inputs_map *= distribution

    def set_postprocess(self, postprocess_func):
        self._post_select = postprocess_func

    @property
    def mode_of_interest_count(self) -> int:
        return self._n_moi

    @property
    def m(self) -> int:
        return self._n_moi + self._n_heralds

    def add(self, mode_mapping, component):
        """
        Add a component (linear or non linear)
        Checks if it's possible:
        - No output is set on used modes
        - Should fail if lock_inputs has been called and the component uses new modes or defines new inputs
        """
        if self._post_select is not None:
            raise RuntimeError("Cannot add any component to a processor with post-process function")

        connector = ModeConnectionResolver(self, component)
        mode_mapping = connector.resolve(mode_mapping)
        if isinstance(component, Processor):
            self._compose_processor(mode_mapping, component)
        elif isinstance(component, AComponent):
            self._add_component(mode_mapping, component)
        else:
            raise RuntimeError(f"Cannot add {type(component)} object to a Processor")
        return self

    @property
    def out_port_names(self):
        result = [''] * self.m
        for port, m_range in self._out_ports.items():
            for m in m_range:
                result[m] = port.name
        return result

    @property
    def in_port_names(self):
        result = [''] * self.m
        for port, m_range in self._in_ports.items():
            for m in m_range:
                result[m] = port.name
        return result

    def _compose_processor(self, mode_mapping, processor):
        # Remove output ports used to connect the new processor
        for i in mode_mapping:
            port = self.get_output_port(i)
            if port is not None:
                del self._out_ports[port]

        # Compute herald positions
        other_herald_pos = list(processor.heralds.keys())
        new_mode_index = self.m
        for pos in other_herald_pos:
            mode_mapping[new_mode_index] = pos
            new_mode_index += 1
            self._n_heralds += 1

        # Add PERM, component, PERM^-1
        perm_modes, perm_component = self._generate_permutation(mode_mapping)
        if perm_component is not None:
            self._components.append((perm_modes, perm_component))
        for pos, c in processor._components:
            pos = [x + min(mode_mapping) for x in pos]
            self._components.append((pos, c))
        if perm_component is not None:
            perm_inv = perm_component.copy()
            perm_inv.inverse(h=True)
            self._components.append((perm_modes, perm_inv))

        # Retrieve ports from the other processor
        for port, port_range in processor._out_ports.items():
            port_mode = list(mode_mapping.keys())[list(mode_mapping.values()).index(port_range[0])]
            if isinstance(port, Herald):
                self._add_herald(port_mode, port.expected, port.user_given_name)
            else:
                self.add_port(port_mode, port, PortLocation.output)

        # Retrieve post process function from the other processor
        if processor._post_select is not None:
            if perm_component is None:
                self._post_select = processor._post_select
            else:
                perm = perm_component.perm_vector
                c_first = perm_modes[0]
                self._post_select = lambda s: processor._post_select(BasicState([s[perm.index(ii) + c_first]
                                                                                 for ii in range(processor.m)]))

    def _add_component(self, mode_mapping, component):
        perm_modes, perm_component = self._generate_permutation(mode_mapping)
        if perm_component is not None:
            self._components.append((perm_modes, perm_component))

        # if isinstance(component, ALinearCircuit) and not isinstance(component, Circuit):
        #     component = Circuit(component.m).add(0, component, merge=False)
        sorted_modes = list(range(min(mode_mapping), min(mode_mapping)+component.m))
        self._components.append((sorted_modes, component))

    @staticmethod
    def _generate_permutation(mode_mapping: Dict[int, int]):
        m_keys = list(mode_mapping.keys())
        min_m = min(m_keys)
        max_m = max(m_keys)
        missing_modes = [x for x in list(range(min_m, max_m+1)) if x not in m_keys]
        for mm in missing_modes:
            mode_mapping[mm] = max(mode_mapping.values()) + 1
        perm_modes = list(range(min_m, min_m+len(mode_mapping)))
        perm_vect = [mode_mapping[i] for i in sorted(mode_mapping.keys())]
        if perm_vect == perm_modes:
            return perm_modes, None  # No need for a permutation, modes are already sorted
        return perm_modes, PERM(perm_vect)

    def _add_herald(self, mode, expected, name=None):
        """
        This internal implementation does not increase the herald count or decrease the mode of interest count
        """
        if not self.are_modes_free([mode], PortLocation.in_out):
            raise UnavailableModeException(mode, "Another port overlaps")
        if name is None:
            name = self._anon_herald_num
            self._anon_herald_num += 1
        self._in_ports[Herald(expected, name)] = [mode]
        self._out_ports[Herald(expected, name)] = [mode]

    def add_herald(self, mode, expected, name=None):
        self._n_moi -= 1
        self._n_heralds += 1
        self._add_herald(mode, expected, name)
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

    def is_mode_connectible(self, mode: int) -> bool:
        if mode < 0:
            return False
        if mode >= self.m:
            return True
        return not self._closed_photonic_modes[mode]

    def are_modes_free(self, mode_range, location: PortLocation = PortLocation.output) -> bool:
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
        return self._post_select(state)
