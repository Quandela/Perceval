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

from abc import ABC, abstractmethod
import copy
from enum import Enum
from typing import Any, Dict, List, Union

from perceval.components.linear_circuit import Circuit, ACircuit
from ._mode_connector import ModeConnector, UnavailableModeException
from perceval.utils import BasicState, StateVector, SVDistribution, Parameter
from .port import LogicalState, Herald, PortLocation, APort
from .abstract_component import AComponent
from .unitary_components import PERM, Unitary
from .non_unitary_components import TD
from .source import Source
from perceval.utils.algorithms.simplification import perm_compose, simplify


class ProcessorType(Enum):
    SIMULATOR = 1
    PHYSICAL = 2


class AProcessor(ABC):
    def __init__(self):
        self._input_state = None
        self.name: str = ""
        self._parameters: Dict = {}

        self._thresholded_output: bool = False
        self._min_detected_photons = None

        self._reset_circuit()

    def _reset_circuit(self):
        self._in_ports: Dict = {}
        self._out_ports: Dict = {}
        self._postprocess = None

        self._is_unitary: bool = True
        self._has_td: bool = False

        self._n_heralds: int = 0
        self._anon_herald_num: int = 0  # This is not a herald count!
        self._components: List[AComponent] = []  # Any type of components, not only unitary ones

        self._n_moi = None  # Number of modes of interest (moi)

    @property
    @abstractmethod
    def type(self) -> ProcessorType:
        pass

    @property
    @abstractmethod
    def is_remote(self) -> bool:
        pass

    @property
    def specs(self):
        return dict()

    def set_parameters(self, params: Dict):
        self._parameters.update(params)

    def set_parameter(self, key: str, value: Any):
        self._parameters[key] = value

    @property
    def parameters(self):
        return self._parameters

    def clear_parameters(self):
        self._parameters = {}

    def clear_input_and_circuit(self):
        self._reset_circuit()
        self._input_state = None

    def min_detected_photons_filter(self, n: int):
        r"""
        Sets-up a state post-selection on the number of detected photons. With thresholded detectors, this will
        actually filter on "click" count.

        :param n: Minimum expected photons

        This post-selection has an impact on the output physical performance
        """
        self.set_parameter('min_detected_photons', n)
        self._min_detected_photons = n

    @property
    def input_state(self):
        return self._input_state

    @property
    @abstractmethod
    def available_commands(self) -> List[str]:
        pass

    def postprocess_output(self, s: BasicState, keep_herald: bool = False) -> BasicState:
        if (not self.heralds or keep_herald) and not self.is_threshold:
            return s
        new_state = []
        for idx, k in enumerate(s):
            if idx in self.heralds:
                continue
            if k > 0 and self.is_threshold:
                k = 1
            new_state.append(k)
        return BasicState(new_state)

    @property
    def post_select_fn(self):
        return self._postprocess

    def set_postprocess(self, postprocess_func):
        r"""
        Set a logical post-selection function. Along with the heralded modes, this function has an impact
        on the logical performance of the processor

        :param postprocess_func: Sets a post-selection function. Its signature must be `func(s: BasicState) -> bool`.
            If None is passed as parameter, removes the previously defined post-selection function.
        """
        self._postprocess = postprocess_func

    def clear_postprocess(self):
        self._postprocess = None

    def _state_selected(self, state: BasicState) -> bool:
        """
        Computes if the state is selected given heralds and post selection function
        """
        for m, v in self.heralds.items():
            if state[m] != v:
                return False
        if self._postprocess is not None:
            return self._postprocess(state)
        return True

    def copy(self, subs: Union[dict, list] = None):
        new_proc = copy.deepcopy(self)
        new_proc._components = []
        for r, c in self._components:
            new_proc._components.append((r, c.copy(subs=subs)))
        return new_proc

    def set_circuit(self, circuit: ACircuit):
        r"""
        Removes all components and replace them by the given circuit's components.
        :return: self. Allows to directly chain this with .add
        """
        if self._n_moi is None:
            self._n_moi = circuit.m
        assert circuit.m == self.circuit_size, "Circuit doesn't have the right number of modes"
        self._components = []
        for r, c in circuit:
            self._components.append((r, c))
        return self

    def add(self, mode_mapping, component, keep_port=True):
        """
        Add a component to the processor (unitary or non-unitary).

        :param mode_mapping: Describe how the new component is connected to the existing processor. Can be:

         * an int: composition uses consecutive modes starting from `mode_mapping`
         * a list or a dict: describes the full mapping of length the input mode count of `component`

        :param component: The component to append to the processor. Can be:

         * A unitary circuit
         * A non-unitary component
         * A processor

        :param keep_port: if True, saves `self`'s output ports on modes impacted by the new component, otherwise removes them.

        Adding a component on non-ordered, non-consecutive modes computes the right permutation (PERM component) which
        fits into the existing processor and the new component.

        Example:

        >>> p = Processor("SLOS", 6)
        >>> p.add(0, BS())  # Modes (0, 1) connected to (0, 1) of the added beam splitter
        >>> p.add([2,5], BS())  # Modes (2, 5) of the processor's output connected to (0, 1) of the added beam splitter
        >>> p.add({2:0, 5:1}, BS())  # Same as above
        """
        if self._postprocess is not None:
            raise RuntimeError("Cannot add any component to a processor with a post-process function. You may remove the post-process function by calling clear_postprocess()")

        self._simulator = None  # Invalidate simulator which will have to be recreated later on
        if self._n_moi is None:
            if isinstance(mode_mapping, int):
                self._n_moi = (component.m if isinstance(component, ACircuit) else component.circuit_size) + mode_mapping
            else:
                self._n_moi = max(mode_mapping) + 1  # max of keys in case of dict
        connector = ModeConnector(self, component, mode_mapping)
        if isinstance(component, AProcessor):
            self._compose_processor(connector, component, keep_port)
        elif isinstance(component, AComponent):
            self._add_component(connector.resolve(), component)
        else:
            raise RuntimeError(f"Cannot add {type(component)} object to a Processor")
        return self

    def _compose_processor(self, connector, processor, keep_port: bool):
        self._is_unitary = self._is_unitary and processor._is_unitary
        self._has_td = self._has_td or processor._has_td
        mode_mapping = connector.resolve()
        if not keep_port:
            # Remove output ports used to connect the new processor
            for i in mode_mapping:
                port = self.get_output_port(i)
                if port is not None:
                    del self._out_ports[port]

        # Compute new herald positions
        n_new_heralds = connector.add_heralded_modes(mode_mapping)
        self._n_heralds += n_new_heralds

        # Add PERM, component, PERM^-1
        perm_modes, perm_component = connector.generate_permutation(mode_mapping)
        new_components = []
        if perm_component is not None:
            if len(self._components) > 0 and isinstance(self._components[-1][1], PERM):
                # Simplify composition by merging two consecutive PERM components
                l_perm_r = self._components[-1][0]
                l_perm_vect = self._components[-1][1].perm_vector
                new_range, new_perm_vect = perm_compose(l_perm_r, l_perm_vect, perm_modes, perm_component.perm_vector)
                new_components.append((new_range, PERM(new_perm_vect)))
                self._components.pop(-1)
            else:
                new_components.append((perm_modes, perm_component))
        for pos, c in processor.components:
            pos = [x + min(mode_mapping) for x in pos]
            new_components.append((pos, c))
        if perm_component is not None:
            perm_inv = perm_component.copy()
            perm_inv.inverse(h=True)
            new_components.append((perm_modes, perm_inv))
        new_components = simplify(new_components, self.circuit_size)
        self._components += new_components

        # Retrieve ports from the other processor
        for port, port_range in processor._out_ports.items():
            port_mode = list(mode_mapping.keys())[list(mode_mapping.values()).index(port_range[0])]
            if isinstance(port, Herald):
                self._add_herald(port_mode, port.expected, port.user_given_name)
            else:
                if self.are_modes_free(range(port_mode, port_mode + port.m)):
                    self.add_port(port_mode, port, PortLocation.OUTPUT)

        # Retrieve post process function from the other processor
        if processor._postprocess is not None:
            if perm_component is None:
                self._postprocess = processor._postprocess
            else:
                perm = perm_component.perm_vector
                c_first = perm_modes[0]
                self._postprocess = lambda s: processor._postprocess([s[perm.index(ii) + c_first]
                                                                      for ii in range(processor.circuit_size)])

    def _add_component(self, mode_mapping, component):
        perm_modes, perm_component = ModeConnector.generate_permutation(mode_mapping)
        if perm_component is not None:
            self._components.append((perm_modes, perm_component))

        sorted_modes = list(range(min(mode_mapping), min(mode_mapping)+component.m))
        self._components.append((sorted_modes, component))
        self._is_unitary = self._is_unitary and isinstance(component, ACircuit)
        self._has_td = self._has_td or isinstance(component, TD)

    def _add_herald(self, mode, expected, name=None):
        """
        This internal implementation neither increases the herald count nor decreases the mode of interest count
        """
        if not self.are_modes_free([mode], PortLocation.IN_OUT):
            raise UnavailableModeException(mode, "Another port overlaps")
        if name is None:
            name = self._anon_herald_num
            self._anon_herald_num += 1
        self._in_ports[Herald(expected, name)] = [mode]
        self._out_ports[Herald(expected, name)] = [mode]

    def add_herald(self, mode: int, expected: int, name: str = None):
        r"""
        Add a heralded mode

        :param mode: Mode index of the herald
        :param expected: number of expected photon as input AND output on the given mode (must be 0 or 1)
        :param name: Herald port name. If none is passed, the name is auto-generated
        """
        assert expected == 0 or expected == 1, "expected must be 0 or 1"
        self._add_herald(mode, expected, name)
        self._n_moi -= 1
        self._n_heralds += 1
        return self

    @property
    def components(self):
        return self._components

    @property
    def m(self) -> int:
        r"""
        :return: Number of modes of interest (MOI) defined in the processor
        """
        return self._n_moi

    @property
    def circuit_size(self) -> int:
        r"""
        :return: Total size of the enclosed circuit (i.e. self.m + heralded mode count)
        """
        return self._n_moi + self._n_heralds

    def linear_circuit(self, flatten: bool = False) -> Circuit:
        """
        Creates a linear circuit from internal components, if all internal components are unitary.
        :param flatten: if True, the component recursive hierarchy is discarded, making the output circuit "flat".
        """
        if not self._is_unitary:
            raise RuntimeError("Cannot retrieve a linear circuit because some components are non-unitary")
        circuit = Circuit(self.circuit_size)
        for component in self._components:
            circuit.add(component[0], component[1], merge=flatten)
        return circuit

    def non_unitary_circuit(self, flatten: bool = False) -> List:
        if self._has_td:  # Inherited from the parent processor in this case
            return self.components

        comp = _flatten(self)
        if flatten:
            return comp

        # Compute the unitaries between the non-unitary components
        new_comp = []
        unitary_circuit = Circuit(self.circuit_size)
        min_r = self.circuit_size
        max_r = 0
        for r, c in comp:
            if isinstance(c, ACircuit):
                unitary_circuit.add(r, c)
                min_r = min(min_r, r[0])
                max_r = max(max_r, r[-1] + 1)
            else:
                if unitary_circuit.ncomponents():
                    new_comp.append((tuple(r_i for r_i in range(min_r, max_r)),
                                    Unitary(unitary_circuit.compute_unitary()[min_r:max_r, min_r:max_r])))
                    unitary_circuit = Circuit(self.circuit_size)
                    min_r = self.circuit_size
                    max_r = 0
                new_comp.append((r, c))

        if unitary_circuit.ncomponents():
            new_comp.append((tuple(r_i for r_i in range(min_r, max_r)),
                             Unitary(unitary_circuit.compute_unitary()[min_r:max_r, min_r:max_r])))

        return new_comp

    def get_circuit_parameters(self) -> Dict[str, Parameter]:
        return {p.name: p for _, c in self._components for p in c.get_parameters()}

    @property
    def out_port_names(self):
        r"""
        :return: A list of the output port names. Names are repeated for ports connected to more than one mode
        """
        result = [''] * self.circuit_size
        for port, m_range in self._out_ports.items():
            for m in m_range:
                result[m] = port.name
        return result

    @property
    def in_port_names(self):
        r"""
        :return: A list of the input port names. Names are repeated for ports connected to more than one mode
        """
        result = [''] * self.circuit_size
        for port, m_range in self._in_ports.items():
            for m in m_range:
                result[m] = port.name
        return result

    def add_port(self, m, port: APort, location: PortLocation = PortLocation.IN_OUT):
        port_range = list(range(m, m + port.m))
        assert port.supports_location(location), f"Port is not compatible with location '{location.name}'"

        if location == PortLocation.IN_OUT or location == PortLocation.INPUT:
            if not self.are_modes_free(port_range, PortLocation.INPUT):
                raise UnavailableModeException(port_range, "Another port overlaps")
            self._in_ports[port] = port_range

        if location == PortLocation.IN_OUT or location == PortLocation.OUTPUT:
            if not self.are_modes_free(port_range, PortLocation.OUTPUT):
                raise UnavailableModeException(port_range, "Another port overlaps")
            self._out_ports[port] = port_range
        return self

    @property
    def _closed_photonic_modes(self):
        output = [False] * self.circuit_size
        for port, m_range in self._out_ports.items():
            if port.is_output_photonic_mode_closed():
                for i in m_range:
                    output[i] = True
        return output

    def is_mode_connectible(self, mode: int) -> bool:
        if mode < 0:
            return False
        if mode >= self.circuit_size:
            return False
        return not self._closed_photonic_modes[mode]

    def are_modes_free(self, mode_range, location: PortLocation = PortLocation.OUTPUT) -> bool:
        """
        :return: True if all modes in mode_range are free of ports, for a given location (input, output or both)
        """
        if location == PortLocation.IN_OUT or location == PortLocation.INPUT:
            for m in mode_range:
                if self.get_input_port(m) is not None:
                    return False
        if location == PortLocation.IN_OUT or location == PortLocation.OUTPUT:
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

    def thresholded_output(self, value: bool):
        r"""
        Simulate threshold detectors on output states. All detections of more than one photon on any given mode is
        changed to 1.

        :param value: enables threshold detection when True, otherwise disables it.
        """
        self._thresholded_output = value

    @property
    def is_threshold(self) -> bool:
        return self._thresholded_output

    @property
    def heralds(self):
        pos = {}
        for port, port_range in self._out_ports.items():
            if isinstance(port, Herald):
                pos[port_range[0]] = port.expected
        return pos

    def _with_logical_input(self, input_state: LogicalState):
        input_state = input_state.to_basic_state(list(self._in_ports.keys()))
        self.with_input(input_state)

    @property
    def source_distribution(self) -> Union[SVDistribution, None]:
        r"""
        Retrieve the computed input distribution.
        :return: the input SVDistribution if `with_input` was called previously, otherwise None.
        """
        return self._inputs_map

    @property
    def source(self):
        r"""
        :return: The photonic source
        """
        return self._source

    @source.setter
    def source(self, source: Source):
        r"""
        :param source: A Source instance to use as the new source for this processor.
        Input distribution is reset when a source is set, so `with_input` has to be called again afterwards.
        """
        self._source = source
        self._inputs_map = None

    def flatten(self) -> List:
        """
        :return: a component list where recursive circuits have been flattened
        """
        return _flatten(self)


def _flatten(composite, starting_mode=0) -> List:
    component_list = []
    for m_range, comp in composite._components:
        if isinstance(comp, Circuit):
            sub_list = _flatten(comp, starting_mode=m_range[0])
            component_list += sub_list
        else:
            m_range = [m + starting_mode for m in m_range]
            component_list.append((m_range, comp))
    return component_list
