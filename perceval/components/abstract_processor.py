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

import copy
from abc import ABC, abstractmethod
from enum import Enum
from multipledispatch import dispatch

from perceval.utils import BasicState, Parameter, PostSelect, LogicalState, NoiseModel, ModeType
from perceval.utils.logging import get_logger, channel, deprecated
from perceval.utils.algorithms.simplification import perm_compose, simplify
from ._mode_connector import ModeConnector, UnavailableModeException
from .abstract_component import AComponent, AParametrizedComponent
from .detector import IDetector, Detector, DetectionType, get_detection_type
from .feed_forward_configurator import AFFConfigurator
from .linear_circuit import Circuit, ACircuit
from .non_unitary_components import TD
from .port import Herald, PortLocation, APort, get_basic_state_from_ports
from .unitary_components import Barrier, PERM, Unitary


class ProcessorType(Enum):
    SIMULATOR = 1
    PHYSICAL = 2


class AProcessor(ABC):
    def __init__(self):
        self._input_state = None
        self.name: str = ""
        self._parameters: dict[str, any] = {}

        self._noise: NoiseModel | None = None

        self._thresholded_output: bool = False  # Deprecated, avoid using this field
        self._min_detected_photons_filter: int | None = None

        self._reset_circuit()

    def _reset_circuit(self):
        self._in_ports: dict = {}
        self._out_ports: dict = {}
        self._postselect: PostSelect | None = None

        self._is_unitary: bool = True
        self._has_td: bool = False
        self._has_feedforward = False

        self._n_heralds: int = 0
        self._anon_herald_num: int = 0  # This is not a herald count!
        self._components: list[tuple[tuple, AComponent]] = []  # Any type of components, not only unitary ones
        self._detectors: list[IDetector] = []
        self.detectors_injected: list[int] = []  # List of modes where detectors are already in the circuit
        self._mode_type: list[ModeType] = []

        self._n_moi: int = 0  # Number of modes of interest (moi)

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

    def set_parameters(self, params: dict[str, any]):
        for key, value in params.items():
            self.set_parameter(key, value)

    def set_parameter(self, key: str, value: any):
        if not isinstance(key, str):
            raise TypeError(f"A parameter name has to be a string (got {type(key)})")
        self._parameters[key] = value

    @property
    def parameters(self):
        return self._parameters

    def clear_parameters(self):
        self._parameters = {}

    def clear_input_and_circuit(self, new_m=None):
        get_logger().debug(f"Clear input and circuit in processor {self.name}", channel.general)
        self._reset_circuit()
        self._input_state = None
        self._circuit_changed()
        if new_m is not None:
            self.m = new_m

    def _circuit_changed(self):
        # Can be used by child class to react to a circuit change
        pass

    def min_detected_photons_filter(self, n: int):
        r"""
        Sets-up a state post-selection on the number of detected photons. With thresholded detectors, this will
        actually filter on "click" count.

        :param n: Minimum expected photons

        This post-selection has an impact on the output physical performance
        """
        self.set_parameter('min_detected_photons', n)
        self._min_detected_photons_filter = n

    @property
    def input_state(self):
        return self._input_state

    @property
    def noise(self):
        return self._noise

    @noise.setter
    def noise(self, nm: NoiseModel):
        if nm is None or isinstance(nm, NoiseModel):
            self._noise = nm
        else:
            raise TypeError("noise type has to be 'NoiseModel'")

    @property
    @abstractmethod
    def available_commands(self) -> list[str]:
        pass

    def remove_heralded_modes(self, s: BasicState) -> BasicState:
        if self.heralds:
            s = s.remove_modes(list(self.heralds.keys()))
        return s

    @property
    def post_select_fn(self):
        return self._postselect

    def set_postselection(self, postselect: PostSelect):
        r"""
        Set a logical post-selection function. Along with the heralded modes, this function has an impact
        on the logical performance of the processor

        :param postselect: Sets a post-selection function. Its signature must be `func(s: BasicState) -> bool`.
            If None is passed as parameter, removes the previously defined post-selection function.
        """
        if not isinstance(postselect, PostSelect):
            raise TypeError("Parameter must be a PostSelect object")
        self._circuit_changed()
        self._postselect = postselect

    def clear_postselection(self):
        if self._postselect is not None:
            self._circuit_changed()
            self._postselect = None

    def _state_selected(self, state: BasicState) -> bool:
        """
        Computes if the state is selected given heralds and post selection function
        """
        for m, v in self.heralds.items():
            if state[m] != v:
                return False
        if self._postselect is not None:
            return self._postselect(state)
        return True

    def copy(self, subs: dict | list = None):
        get_logger().debug(f"Copy processor {self.name}", channel.general)
        new_proc = copy.copy(self)
        new_proc._components = []
        for r, c in self._components:
            new_proc._components.append((r, c.copy(subs=subs)))
        return new_proc

    def set_circuit(self, circuit: ACircuit) -> AProcessor:
        r"""
        Removes all components and replace them by the given circuit.

        :param circuit: The circuit to start the processor with
        :return: Self to allow direct chain this with .add()
        """
        if self._n_moi == 0:
            self.m = circuit.m
        assert circuit.m == self.circuit_size, "Circuit doesn't have the right number of modes"
        self._components = []
        for r, c in circuit:
            self._components.append((r, c))
        return self

    def add(self, mode_mapping, component, keep_port: bool = True) -> AProcessor:
        """
        Add a component to the processor (unitary or non-unitary).

        :param mode_mapping: Describe how the new component is connected to the existing processor. Can be:

         * an int: composition uses consecutive modes starting from `mode_mapping`
         * a list or a dict: describes the full mapping of length the input mode count of `component`

        :param component: The component to append to the processor. Can be:

         * A unitary circuit
         * A non-unitary component
         * A processor
         * A detector

        :param keep_port: if True, saves `self`'s output ports on modes impacted by the new component, otherwise removes them.

        Adding a component on non-ordered, non-consecutive modes computes the right permutation (PERM component) which
        fits into the existing processor and the new component.

        Example:

        >>> p = Processor("SLOS", 6)
        >>> p.add(0, BS())  # Modes (0, 1) connected to (0, 1) of the added beam splitter
        >>> p.add([2,5], BS())  # Modes (2, 5) of the processor's output connected to (0, 1) of the added beam splitter
        >>> p.add({2:0, 5:1}, BS())  # Same as above
        """
        if self.m == 0:
            self.m = component.m + mode_mapping if isinstance(mode_mapping, int) else max(mode_mapping) + 1
            get_logger().debug(f"Number of modes of interest defaulted to {self.m} in processor {self.name}",
                               channel.general)

        connector = ModeConnector(self, component, mode_mapping)
        if isinstance(component, AProcessor):
            self._compose_processor(connector, component, keep_port)
        elif isinstance(component, IDetector):
            self._add_detector(mode_mapping, component)
        elif isinstance(component, AFFConfigurator):
            self._add_ffconfig(mode_mapping, component)
        elif isinstance(component, Barrier):
            self._components.append((mode_mapping, component))
        elif isinstance(component, AComponent):
            self._add_component(connector.resolve(), component, keep_port)
        else:
            raise RuntimeError(f"Cannot add {type(component)} object to a Processor")
        self._circuit_changed()
        return self

    def _add_ffconfig(self, modes, component: AFFConfigurator):
        if isinstance(modes, int):
            modes = tuple(range(modes, modes + component.m))

        # Check composition consistency
        if min(modes) < 0 or max(modes) >= self.m:
            raise ValueError(f"Mode numbers must be in [0; {self.m - 1}] (got {modes})")
        if any([self._mode_type[i] != ModeType.CLASSICAL for i in modes]):
            raise UnavailableModeException(modes, "Cannot add a classical component on non-classical modes")
        photonic_modes = component.config_modes(modes)
        if min(photonic_modes) < 0 or max(photonic_modes) >= self.m:
            raise ValueError(f"Mode numbers must be in [0; {self.m - 1}] (got {photonic_modes})")
        if any([self._mode_type[i] != ModeType.PHOTONIC for i in photonic_modes]):
            raise UnavailableModeException(photonic_modes, "Cannot add a configured circuit on non-photonic modes")

        modes_add_detectors = [m for m in modes if m not in self.detectors_injected]
        self._components.append((tuple(range(self.m)), Barrier(self.m, visible=len(modes_add_detectors) == 0)))
        if modes_add_detectors and modes_add_detectors[0] > 0:  # Barrier above detectors
            ports = tuple(range(0, modes_add_detectors[0]))
            self._components.append((ports, Barrier(len(ports), visible=True)))
        for m in modes_add_detectors:
            self.detectors_injected.append(m)
            self._components.append(((m,), self._detectors[m]))
        if modes_add_detectors and modes_add_detectors[-1] < self.m - 1:  # Barrier below detectors
            ports = tuple(range(modes_add_detectors[-1] + 1, self.m))
            self._components.append((ports, Barrier(len(ports), visible=True)))
        self._components.append((modes, component))
        self._has_feedforward = True
        self._is_unitary = False
        component.block_circuit_size()  # User cannot add larger photonic circuit output from now on

    def _add_detector(self, mode: int, detector: IDetector):
        if isinstance(mode, (tuple, list)) and len(mode) == 1:
            mode = mode[0]

        if not isinstance(mode, int):
            raise TypeError(f"When adding a detector, the mode number must be an integer (got {type(mode)})")

        if self._mode_type[mode] == ModeType.CLASSICAL:
            raise UnavailableModeException(mode, "Mode is not photonic, cannot plug a detector.")
        self._detectors[mode] = detector
        if self._mode_type[mode] == ModeType.PHOTONIC:
            self._mode_type[mode] = ModeType.CLASSICAL

    @property
    def detectors(self):
        return self._detectors

    def _validate_postselect_composition(self, mode_mapping: dict):
        if self._postselect is not None and isinstance(self._postselect, PostSelect):
            impacted_modes = list(mode_mapping.keys())
            # can_compose_with can take a bit of time so leave this test as an assert which can be removed by -O
            assert self._postselect.can_compose_with(impacted_modes), \
                f"Post-selection conditions cannot compose with modes {impacted_modes}"

    def _compose_processor(self, connector: ModeConnector, processor, keep_port: bool):
        get_logger().debug(f"Compose processor {self.name} with {processor.name}", channel.general)
        self._is_unitary = self._is_unitary and processor._is_unitary
        self._has_td = self._has_td or processor._has_td
        if processor.heralds and not processor.parameters:
            # adding the same processor component again renders incorrect heralds if not copied
            # This concerns our gate based processors from catalog which has no input params
            get_logger().debug("  Force copy during processor compose", channel.general)
            processor = processor.copy()

        mode_mapping = connector.resolve()
        get_logger().debug(f"  Resolved mode mapping to {mode_mapping} during processor compose", channel.general)

        self._validate_postselect_composition(mode_mapping)
        if not keep_port:
            # Remove output ports used to connect the new processor
            for i in mode_mapping:
                port = self.get_output_port(i)
                if port is not None:
                    del self._out_ports[port]

        # Compute new herald positions
        n_new_heralds = connector.add_heralded_modes(mode_mapping)
        self._n_heralds += n_new_heralds
        self._mode_type += [ModeType.HERALD] * n_new_heralds
        for m_herald in processor.heralds:
            self._detectors += [processor._detectors[m_herald]]

        # Add PERM, component, PERM^-1
        perm_modes, perm_component = connector.generate_permutation(mode_mapping)
        new_components = []
        if perm_component is not None:
            get_logger().debug(
                f"  Add {perm_component.perm_vector} permutation before processor compose", channel.general)
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
            get_logger().debug(f"  Add {perm_inv.perm_vector} permutation after processor compose", channel.general)
            new_components.append((perm_modes, perm_inv))
        new_components = simplify(new_components, self.circuit_size)
        self._components += new_components

        # Retrieve ports from the other processor
        # Output ports
        for port, port_range in processor._out_ports.items():
            port_mode = list(mode_mapping.keys())[list(mode_mapping.values()).index(port_range[0])]
            if isinstance(port, Herald):
                self._add_herald(port_mode, port.expected, port.user_given_name)
            else:
                if self.are_modes_free(range(port_mode, port_mode + port.m)):
                    self.add_port(port_mode, port, PortLocation.OUTPUT)
        # Input ports
        for port, port_range in processor._in_ports.items():
            port_mode = list(mode_mapping.keys())[list(mode_mapping.values()).index(port_range[0])]
            if self.are_modes_free(range(port_mode, port_mode + port.m), PortLocation.INPUT):
                self.add_port(port_mode, port, PortLocation.INPUT)

        # Retrieve post process function from the other processor
        if processor._postselect is not None:
            c_first = perm_modes[0]
            other_postselect = copy.copy(processor._postselect)
            if perm_component is not None:
                other_postselect.apply_permutation(perm_inv.perm_vector, c_first)
            other_postselect.shift_modes(c_first)
            if not (self._postselect is None or other_postselect is None
                    or self._postselect.is_independent_with(other_postselect)):
                raise RuntimeError("Cannot automatically compose processor's post-selection conditions")
            self._postselect = self._postselect or PostSelect()
            self._postselect.merge(other_postselect)

    def _add_component(self, mode_mapping, component, keep_port: bool):
        self._validate_postselect_composition(mode_mapping)
        if not keep_port:
            # Remove output ports used to connect the new processor
            for i in mode_mapping:
                port = self.get_output_port(i)
                if port is not None:
                    del self._out_ports[port]

        perm_modes, perm_component = ModeConnector.generate_permutation(mode_mapping)
        if perm_component is not None:
            self._components.append((perm_modes, perm_component))

        sorted_modes = tuple(range(min(mode_mapping), min(mode_mapping)+component.m))
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
        self._mode_type[mode] = ModeType.HERALD
        self._circuit_changed()

    def add_herald(self, mode: int, expected: int, name: str = None) -> AProcessor:
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
        """
        :return: Number of modes of interest (MOI) defined in the processor
        """
        return self._n_moi

    @m.setter
    def m(self, value: int):
        if self._n_moi != 0:
            raise RuntimeError(f"The number of modes of this processor was already set (to {self._n_moi})")
        if not isinstance(value, int) or value < 1:
            raise ValueError(f"The number of modes should be a strictly positive integer (got {value})")
        self._n_moi = value
        self._detectors = [None] * value
        self._mode_type = [ModeType.PHOTONIC] * value

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
        for pos_m, component in self._components:
            circuit.add(pos_m, component, merge=flatten)
        return circuit

    def non_unitary_circuit(self, flatten: bool = False) -> list[tuple[tuple, AComponent]]:
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

    def get_circuit_parameters(self) -> dict[str, Parameter]:
        return {p.name: p for _, c in self._components if isinstance(c, AParametrizedComponent)
                for p in c.get_parameters()}

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

    @staticmethod
    def _find_and_remove_port_from_list(m, port_list) -> bool:
        for current_port, current_port_range in port_list.items():
            if m in current_port_range:
                del port_list[current_port]
                return True
        return False

    def remove_port(self, m, location: PortLocation = PortLocation.IN_OUT):
        if location in (PortLocation.IN_OUT, PortLocation.INPUT):
            if not AProcessor._find_and_remove_port_from_list(m, self._in_ports):
                raise UnavailableModeException(m, f"Port is not at location '{location.name}'")

        if location in (PortLocation.IN_OUT, PortLocation.OUTPUT):
            if not AProcessor._find_and_remove_port_from_list(m, self._out_ports):
                raise UnavailableModeException(m, f"Port is not at location '{location.name}'")
        return self

    def is_mode_connectible(self, mode: int) -> bool:
        if mode < 0:
            return False
        if mode >= self.circuit_size:
            return False
        return self._mode_type[mode] == ModeType.PHOTONIC

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

    @deprecated(version=0.12, reason="Add detectors as components")
    def thresholded_output(self, value: bool):
        r"""
        Simulate threshold detectors on output states. All detections of more than one photon on any given mode is
        changed to 1.

        :param value: enables threshold detection when True, otherwise disables it.
        """
        self._thresholded_output = value
        self._detectors = [Detector.threshold() if self._thresholded_output else None] * len(self._detectors)

    @property
    @deprecated(version=0.12, reason="Use `detection_type` property instead")
    def is_threshold(self) -> bool:
        return self._thresholded_output

    @property
    def detection_type(self) -> DetectionType:
        return get_detection_type(self._detectors)

    @property
    def heralds(self):
        pos = {}
        for port, port_range in self._out_ports.items():
            if isinstance(port, Herald):
                pos[port_range[0]] = port.expected
        return pos

    def _with_logical_input(self, input_state: LogicalState):
        input_state = get_basic_state_from_ports(list(self._in_ports.keys()), input_state)
        self.with_input(input_state)

    def check_input(self, input_state: BasicState):
        r"""Check if a basic state input matches with the current processor configuration"""
        assert self.m is not None, "A circuit has to be set before the input state"
        expected_input_length = self.m
        assert len(input_state) == expected_input_length, \
            f"Input length not compatible with circuit (expects {expected_input_length}, got {len(input_state)})"
        if input_state.has_polarization:
            get_logger().warn("Given input state has polarization, that will be ignored in the computation"
                              " (use with_polarized_input instead).")
        elif input_state.has_annotations:
            get_logger().warn("Given input state has annotations, that will be ignored in the computation."
                              " To use them, consider using a StateVector.")

    def _deduce_min_detected_photons(self, expected_photons: int) -> None:
        get_logger().warn(
            "Setting a value for min_detected_photons will soon be mandatory, please change your scripts accordingly." +
            " Use the method processor.min_detected_photons_filter(value) before any call of processor.with_input(input)." +
            f" The current deduced value of min_detected_photons is {expected_photons}", channel.user)
        self._min_detected_photons_filter = expected_photons

    @dispatch(BasicState)
    def with_input(self, input_state: BasicState) -> None:
        self.check_input(input_state)
        input_list = [0] * self.circuit_size
        input_idx = 0
        expected_photons = 0
        # Build real input state (merging ancillas + expected input) and compute expected photon count
        for k in range(self.circuit_size):
            if k in self.heralds:
                input_list[k] = self.heralds[k]
                expected_photons += self.heralds[k]
            else:
                input_list[k] = input_state[input_idx]
                expected_photons += input_state[input_idx]
                input_idx += 1

        self._input_state = BasicState(input_list)

        if self._min_detected_photons_filter is None:
            self._deduce_min_detected_photons(expected_photons)

    def flatten(self) -> list[tuple]:
        """
        :return: a component list where recursive circuits have been flattened
        """
        return _flatten(self)


def _flatten(composite, starting_mode=0) -> list[tuple]:
    component_list = []
    for m_range, comp in composite._components:
        if isinstance(comp, Circuit):
            sub_list = _flatten(comp, starting_mode=m_range[0])
            component_list += sub_list
        else:
            m_range = [m + starting_mode for m in m_range]
            component_list.append((m_range, comp))
    return component_list
