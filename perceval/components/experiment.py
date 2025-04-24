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
from dataclasses import dataclass

from multipledispatch import dispatch
from numpy import inf

from perceval.utils import BasicState, Parameter, PostSelect, LogicalState, NoiseModel, ModeType, StateVector, \
    SVDistribution
from perceval.utils.logging import get_logger, channel, deprecated
from perceval.utils.algorithms.simplification import perm_compose, simplify
from ._mode_connector import ModeConnector, UnavailableModeException
from .abstract_component import AComponent, AParametrizedComponent
from .detector import IDetector, Detector, DetectionType, get_detection_type
from .feed_forward_configurator import AFFConfigurator
from .linear_circuit import Circuit, ACircuit
from .non_unitary_components import TD
from .port import Herald, PortLocation, APort, get_basic_state_from_ports
from .unitary_components import Barrier, PERM, Unitary, PS


@dataclass
class _PhaseNoise:
    quantization: float = 0
    max_error: float = 0


class Experiment:
    """
    This class represents an optical table containing:

    - A circuit and/or components that represent the operations that will operate on photons.
        Can contain non-unitary components
    - The input state for the experiment.
    - Detectors to detect photons.
    - Ports to define groups of modes
    - Heralds
    - A post-selection method
    - A NoiseModel.
    """

    def __init__(self, m_circuit: int | ACircuit = None, noise: NoiseModel = None, name: str = "Experiment"):
        self._input_state = None
        self.name: str = name

        self._thresholded_output: bool = False  # Deprecated, avoid using this field
        self._min_detected_photons_filter: int | None = None

        self._circuit_changed_observers: list[callable] = []
        self._noise_changed_observers: list[callable] = []
        self._input_changed_observers: list[callable] = []

        self.noise: NoiseModel | None = noise

        self._reset_circuit()
        self._init_circuit(m_circuit)

    def _reset_circuit(self):
        self._in_ports: dict = {}
        self._out_ports: dict = {}
        self._postselect: PostSelect | None = None

        self._is_unitary: bool = True
        self._has_td: bool = False
        self._has_feedforward = False

        self._anon_herald_num: int = 0  # This is not a herald count!
        self._components: list[tuple[tuple, AComponent]] = []  # Any type of components, not only unitary ones
        self._detectors: list[IDetector] = []
        self.detectors_injected: list[int] = []  # List of modes where detectors are already in the circuit
        self._in_mode_type: list[ModeType] = []
        self._out_mode_type: list[ModeType] = []

        self._m: int = 0  # Circuit size

    @property
    def is_unitary(self) -> bool:
        return self._is_unitary

    @property
    def has_td(self) -> bool:
        return self._has_td

    @property
    def has_feedforward(self) -> bool:
        return self._has_feedforward

    def _init_circuit(self, m_circuit: ACircuit | int):
        if isinstance(m_circuit, ACircuit):
            self.m = m_circuit.m
            self.add(0, m_circuit)
        elif m_circuit is not None:
            self.m = m_circuit  # number of modes

    def clear_input_and_circuit(self, new_m=None):
        get_logger().debug(f"Clear input and circuit in experiment {self.name}", channel.general)
        self._reset_circuit()
        self._input_state = None
        self._input_changed()
        self._circuit_changed()
        if new_m is not None:
            self.m = new_m

    def _circuit_changed(self, component=None):
        for observer in self._circuit_changed_observers:
            observer(component)  # Used to notify the Processors containing this experiment of a new component

    def add_observers(self, circuit_observer: callable, noise_observer: callable, input_observer: callable):
        self._circuit_changed_observers.append(circuit_observer)
        self._noise_changed_observers.append(noise_observer)
        self._input_changed_observers.append(input_observer)

    def min_detected_photons_filter(self, n: int):
        r"""
        Sets-up a state post-selection on the number of detected photons. With thresholded detectors, this will
        actually filter on "click" count.

        :param n: Minimum expected photons

        This post-selection has an impact on the output physical performance
        """
        self._min_detected_photons_filter = n

    @property
    def min_photons_filter(self):
        return self._min_detected_photons_filter

    @property
    def input_state(self):
        return self._input_state

    @property
    def noise(self):
        return self._noise

    @noise.setter
    def noise(self, nm: NoiseModel | None):
        if nm is not None:
            self._phase_noise = _PhaseNoise(nm.phase_imprecision, nm.phase_error)
        else:
            self._phase_noise = _PhaseNoise()
        if nm is None or isinstance(nm, NoiseModel):
            self._noise = nm
            for observer in self._noise_changed_observers:
                observer()
        else:
            raise TypeError("noise type has to be 'NoiseModel'")

    @property
    def post_select_fn(self):
        return self._postselect

    def set_postselection(self, postselect: PostSelect):
        r"""
        Set a logical post-selection function. Along with the heralded modes, this function has an impact
        on the logical performance of the processor holding this experiment

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

    def copy(self, subs: dict | list = None):
        get_logger().debug(f"Copy experiment {self.name}", channel.general)
        new_proc = copy.copy(self)
        new_proc._components = []
        for r, c in self._components:
            new_proc._components.append((r, c.copy(subs=subs)))
        return new_proc

    def set_circuit(self, circuit: ACircuit):
        r"""
        Removes all components and replace them by the given circuit.

        :param circuit: The circuit to start the experiment with
        :return: Self to allow direct chain this with .add()
        """
        if self._m == 0:
            self.m = circuit.m
        assert circuit.m == self.circuit_size, "Circuit doesn't have the right number of modes"
        self._components = []
        for r, c in circuit:
            self._components.append((r, c))
        return self

    def add(self, mode_mapping, component, keep_port: bool = True):
        """
        Add a component to the experiment (unitary or non-unitary).

        :param mode_mapping: Describe how the new component is connected to the existing experiment. Can be:

         * an int: composition uses consecutive modes starting from `mode_mapping`
         * a list or a dict: describes the full mapping of length the input mode count of `component`

        :param component: The component to append to the experiment. Can be:

         * A unitary circuit
         * A non-unitary component
         * A processor
         * An experiment
         * A detector

        :param keep_port: if True, saves `self`'s output ports on modes impacted by the new component, otherwise removes them.

        Adding a component on non-ordered, non-consecutive modes computes the right permutation (PERM component) which
        fits into the existing experiment and the new component.

        Example:

        >>> e = Experiment(6)
        >>> e.add(0, BS())  # Modes (0, 1) connected to (0, 1) of the added beam splitter
        >>> e.add([2,5], BS())  # Modes (2, 5) of the experiment's output connected to (0, 1) of the added beam splitter
        >>> e.add({2:0, 5:1}, BS())  # Same as above

        If the added component is a processor or an experiment with modes having heralds only on one side,
        no permutation will be added at the end, and the "in-between" modes will be pushed to the bottom.
        """
        if self.m == 0:
            self.m = component.m + mode_mapping if isinstance(mode_mapping, int) else max(mode_mapping) + 1
            get_logger().debug(f"Number of modes of interest defaulted to {self.m} in experiment {self.name}",
                               channel.general)

        from perceval import AProcessor  # This is ugly but necessary to keep the user interface
        if isinstance(component, AProcessor):
            component = component.experiment

        connector = ModeConnector(self, component, mode_mapping)

        if isinstance(component, Experiment):
            self._compose_experiment(connector, component, keep_port)
        elif isinstance(component, IDetector):
            self._add_detector(mode_mapping, component)
        elif isinstance(component, AFFConfigurator):
            self._add_ffconfig(mode_mapping, component)
        elif isinstance(component, Barrier):
            if isinstance(mode_mapping, int):
                mode_mapping = tuple(range(mode_mapping, mode_mapping + component.m))
            self._components.append((mode_mapping, component))
        elif isinstance(component, AComponent):
            self._add_component(connector.resolve(), component, keep_port)
        else:
            raise RuntimeError(f"Cannot add {type(component)} object to a Processor")

        self._circuit_changed(component)
        return self

    def _add_ffconfig(self, modes, component: AFFConfigurator):
        if isinstance(modes, int):
            modes = tuple(range(modes, modes + component.m))

        # Check composition consistency
        if min(modes) < 0 or max(modes) >= self.m:
            raise ValueError(f"Mode numbers must be in [0; {self.m - 1}] (got {modes})")
        if any([self._out_mode_type[i] != ModeType.CLASSICAL for i in modes]):
            raise UnavailableModeException(modes, "Cannot add a classical component on non-classical modes")
        photonic_modes = component.config_modes(modes)
        if min(photonic_modes) < 0 or max(photonic_modes) >= self.m:
            raise ValueError(f"Mode numbers must be in [0; {self.m - 1}] (got {photonic_modes})")
        if any([self._out_mode_type[i] != ModeType.PHOTONIC for i in photonic_modes]):
            raise UnavailableModeException(photonic_modes, "Cannot add a configured circuit on non-photonic modes")

        modes_add_detectors = [m for m in modes if m not in self.detectors_injected]
        self._components.append((tuple(range(self.m)), Barrier(self.m, visible=len(modes_add_detectors) == 0)))
        if modes_add_detectors and modes_add_detectors[0] > 0:  # Barrier above detectors
            ports = tuple(range(0, modes_add_detectors[0]))
            self._components.append((ports, Barrier(len(ports), visible=True)))
        if modes_add_detectors and modes_add_detectors[-1] < self.m - 1:  # Barrier below detectors
            ports = tuple(range(modes_add_detectors[-1] + 1, self.m))
            self._components.append((ports, Barrier(len(ports), visible=True)))
        for m in modes_add_detectors:
            self.detectors_injected.append(m)
            self._components.append(((m,), self._detectors[m]))
        self._components.append((modes, component))
        self._has_feedforward = True
        self._is_unitary = False
        component.block_circuit_size()  # User cannot add larger photonic circuit output from now on

    def _add_detector(self, mode: int, detector: IDetector):
        if isinstance(mode, (tuple, list)) and len(mode) == 1:
            mode = mode[0]

        if not isinstance(mode, int):
            raise TypeError(f"When adding a detector, the mode number must be an integer (got {type(mode)})")

        if self._out_mode_type[mode] == ModeType.CLASSICAL:
            raise UnavailableModeException(mode, "Mode is not photonic, cannot plug a detector.")
        self._detectors[mode] = detector
        if self._out_mode_type[mode] == ModeType.PHOTONIC:
            self._out_mode_type[mode] = ModeType.CLASSICAL

    @property
    def detectors(self):
        return self._detectors

    def _validate_postselect_composition(self, mode_mapping: dict):
        if self._postselect is not None and isinstance(self._postselect, PostSelect):
            impacted_modes = list(mode_mapping.keys())
            # can_compose_with can take a bit of time so leave this test as an assert which can be removed by -O
            assert self._postselect.can_compose_with(impacted_modes), \
                f"Post-selection conditions cannot compose with modes {impacted_modes}"

    def _compose_experiment(self, connector: ModeConnector, experiment: Experiment, keep_port: bool):
        get_logger().debug(f"Compose experiment {self.name} with {experiment.name}", channel.general)
        self._is_unitary = self._is_unitary and experiment._is_unitary
        self._has_td = self._has_td or experiment._has_td
        if experiment.heralds:
            # adding the same experiment component again renders incorrect heralds if not copied
            # This concerns our gate based processors from catalog which has no input params
            get_logger().debug("  Force copy during experiment compose", channel.general)
            experiment = experiment.copy()

        mode_mapping = connector.resolve()
        get_logger().debug(f"  Resolved mode mapping to {mode_mapping} during experiment compose", channel.general)

        is_symmetrical = experiment.in_heralds.keys() == experiment.heralds.keys()

        # Compute new herald positions
        n_new_heralds = connector.add_heralded_modes(mode_mapping)

        self._validate_postselect_composition(mode_mapping)
        if not keep_port:
            # Remove output ports used to connect the new experiment
            for i in mode_mapping:
                port = self.get_output_port(i)
                if port is not None:
                    del self._out_ports[port]

        self._in_mode_type += [ModeType.HERALD] * n_new_heralds  # New input heralds are always put at the bottom
        self._m += n_new_heralds

        if is_symmetrical:
            self._out_mode_type += [ModeType.HERALD] * n_new_heralds
            for m_herald in experiment.heralds:
                self._detectors += [experiment._detectors[m_herald]]

        # Check port composition
        for m_out, m_in in mode_mapping.items():
            out_port = self.get_output_port(m_out)
            in_port = experiment.get_input_port(m_in)
            if (out_port is not None and in_port is not None
                    and (out_port.encoding != in_port.encoding or
                         [mode_mapping.get(i, i) for i in self._out_ports[out_port]] != experiment._in_ports[in_port])):
                get_logger().warn(
                    f"The composition of {self.name} ({out_port.encoding} on modes {self._out_ports[out_port]}) "
                    f"with {experiment.name} ({in_port.encoding} on modes {experiment._in_ports[in_port]}) "
                    f"will lead to unexpected results.")
                break

        # Add PERM, component, (PERM^-1 if is_symmetrical)
        perm_modes, perm_component = connector.generate_permutation(mode_mapping)
        new_components = []
        if perm_component is not None:
            get_logger().debug(
                f"  Add {perm_component.perm_vector} permutation before experiment compose", channel.general)
            if len(self._components) > 0 and isinstance(self._components[-1][1], PERM):
                # Simplify composition by merging two consecutive PERM components
                l_perm_r = self._components[-1][0]
                l_perm_vect = self._components[-1][1].perm_vector
                new_range, new_perm_vect = perm_compose(l_perm_r, l_perm_vect, perm_modes, perm_component.perm_vector)
                new_components.append((new_range, PERM(new_perm_vect)))
                self._components.pop(-1)
            else:
                new_components.append((perm_modes, perm_component))
        for pos, c in experiment.components:
            pos = [x + min(mode_mapping) for x in pos]
            new_components.append((pos, c))
        if perm_component is not None and is_symmetrical:
            perm_inv = perm_component.copy()
            perm_inv.inverse(h=True)
            get_logger().debug(f"  Add {perm_inv.perm_vector} permutation after experiment compose", channel.general)
            new_components.append((perm_modes, perm_inv))
        elif not is_symmetrical:
            # We need to apply the permutation on the detectors and mode types
            self._out_mode_type = connector.compose_lists(mode_mapping, self._out_mode_type, experiment._out_mode_type)
            self._detectors = connector.compose_lists(mode_mapping, self._detectors, experiment._detectors)

            self_ports = [None] * self.circuit_size
            for port, port_range in self._out_ports.items():
                self_ports[port_range[0]] = port

            other_ports = [None] * experiment.circuit_size
            for port, port_range in experiment._out_ports.items():
                other_ports[port_range[0]] = port

            self._out_ports = {}
            out_ports = connector.compose_lists(mode_mapping, self_ports, other_ports)
            for port_mode, port in enumerate(out_ports):
                if isinstance(port, Herald):
                    self.add_herald(port_mode, port.expected, port.user_given_name, PortLocation.OUTPUT)
                elif port is not None:
                    if self.are_modes_free(range(port_mode, port_mode + port.m)):
                        self.add_port(port_mode, port, PortLocation.OUTPUT)

        new_components = simplify(new_components, self.circuit_size)
        self._components += new_components

        # Retrieve ports from the other experiment
        # Output ports
        if is_symmetrical:
            for port, port_range in experiment._out_ports.items():
                port_mode = list(mode_mapping.keys())[list(mode_mapping.values()).index(port_range[0])]
                if isinstance(port, Herald):
                    self.add_herald(port_mode, port.expected, port.user_given_name, PortLocation.OUTPUT)
                else:
                    if self.are_modes_free(range(port_mode, port_mode + port.m)):
                        self.add_port(port_mode, port, PortLocation.OUTPUT)

        # Input ports
        for port, port_range in experiment._in_ports.items():
            port_mode = list(mode_mapping.keys())[list(mode_mapping.values()).index(port_range[0])]
            if isinstance(port, Herald):
                self.add_herald(port_mode, port.expected, port.user_given_name, PortLocation.INPUT)
            else:
                if self.are_modes_free(range(port_mode, port_mode + port.m), PortLocation.INPUT):
                    self.add_port(port_mode, port, PortLocation.INPUT)

        # Detectors
        if is_symmetrical:
            for m in range(experiment.circuit_size):
                # The heralded modes detectors have already been added at the bottom
                d = experiment.detectors[m]
                if m not in experiment.heralds and d is not None:
                    new_mode = list(mode_mapping.keys())[list(mode_mapping.values()).index(m)]
                    self._detectors[new_mode] = d

        if self._postselect is not None and perm_component is not None and not is_symmetrical:
            c_first = perm_modes[0]
            self._postselect.apply_permutation(perm_component.perm_vector, c_first)

        # Retrieve post process function from the other experiment
        if experiment._postselect is not None:
            c_first = perm_modes[0]
            other_postselect = copy.copy(experiment._postselect)
            if perm_component is not None and is_symmetrical:
                other_postselect.apply_permutation(perm_inv.perm_vector, c_first)
            other_postselect.shift_modes(c_first)
            if not (self._postselect is None or other_postselect is None
                    or self._postselect.is_independent_with(other_postselect)):
                raise RuntimeError("Cannot automatically compose experiment's post-selection conditions")
            self._postselect = self._postselect or PostSelect()
            self._postselect.merge(other_postselect)

    def _add_component(self, mode_mapping, component, keep_port: bool):
        self._validate_postselect_composition(mode_mapping)
        if not keep_port:
            # Remove output ports used to connect the new experiment
            for i in mode_mapping:
                port = self.get_output_port(i)
                if port is not None:
                    del self._out_ports[port]

        perm_modes, perm_component = ModeConnector.generate_permutation(mode_mapping)
        if perm_component is not None:
            self._components.append((perm_modes, perm_component))

        sorted_modes = tuple(range(min(mode_mapping), min(mode_mapping) + component.m))
        self._components.append((sorted_modes, component))
        self._is_unitary = self._is_unitary and isinstance(component, ACircuit)
        self._has_td = self._has_td or isinstance(component, TD)

    def _add_herald(self, mode, herald: Herald, location: PortLocation = PortLocation.IN_OUT):
        if location == PortLocation.INPUT or location == PortLocation.IN_OUT:
            self._in_ports[herald] = [mode]
            self._in_mode_type[mode] = ModeType.HERALD

        if location == PortLocation.OUTPUT or location == PortLocation.IN_OUT:
            self._out_ports[herald] = [mode]
            self._out_mode_type[mode] = ModeType.HERALD
        self._circuit_changed()

    def add_herald(self, mode: int, expected: int, name: str = None, location: PortLocation = PortLocation.IN_OUT):
        r"""
        Add a heralded mode

        :param mode: Mode index of the herald
        :param expected: number of expected photon as input AND output on the given mode (must be 0 or 1)
        :param name: Herald port name. If none is passed, the name is auto-generated
        :param location: Port location of the herald (input, output or both)
        """
        if not self.are_modes_free([mode], location):
            raise UnavailableModeException(mode, "Another port overlaps")

        if name is None:
            name = self._anon_herald_num
            self._anon_herald_num += 1

        herald = Herald(expected, name)
        self._add_herald(mode, herald, location)
        return self

    @property
    def components(self):
        return self._components

    @property
    def m(self) -> int:
        """
        :return: Number of modes of interest (MOI) defined in the experiment for the output
        """
        return self._m - len(self.heralds)

    @property
    def m_in(self):
        """
        :return: Number of modes of interest (MOI) defined in the experiment for the input
        """
        return self._m - len(self.in_heralds)

    @m.setter
    def m(self, value: int):
        """This is actually a setter for the circuit size"""
        if self._m != 0:
            raise RuntimeError(f"The number of modes of this experiment was already set (to {self._m})")
        if not isinstance(value, int) or value < 1:
            raise ValueError(f"The number of modes should be a strictly positive integer (got {value})")
        self._m = value
        self._detectors = [None] * value
        self._in_mode_type = [ModeType.PHOTONIC] * value
        self._out_mode_type = [ModeType.PHOTONIC] * value

    @property
    def circuit_size(self) -> int:
        r"""
        :return: Total size of the enclosed circuit (i.e. self.m + heralded mode count)
        """
        return self._m

    def unitary_circuit(self, flatten: bool = False, use_phase_noise=False) -> Circuit:
        """
        Creates a unitary circuit from internal components, if all internal components are unitary.
        :param flatten: if True, the component recursive hierarchy is discarded, making the output circuit "flat".
        """
        if not self._is_unitary:
            raise RuntimeError("Cannot retrieve a unitary circuit because some components are non-unitary")
        circuit = Circuit(self.circuit_size)
        for pos_m, component in self._components:
            circuit.add(pos_m, component, merge=flatten)
        noise = self._phase_noise
        if not use_phase_noise or not (noise.max_error or noise.quantization):
            return circuit
        # Apply phase quantization noise on all phase parameters in the circuit
        get_logger().debug(f"Inject {noise} in the circuit")
        circuit = circuit.copy()  # Copy the whole circuit in order to keep the initial phase values in self
        for _, component in circuit:
            if not isinstance(component, PS):
                continue
            if noise.max_error is not None:
                err_param = component.param("max_error")
                if not err_param.is_variable and float(err_param) == 0:
                    err_param.set_value(noise.max_error, force=True)
            if noise.quantization:
                phi_param = component.param("phi")
                phi_param.set_value(noise.quantization * round(float(phi_param) / noise.quantization), force=True)
        return circuit

    def non_unitary_circuit(self, flatten: bool = False) -> list[tuple[tuple, AComponent]]:
        if self._has_td:  # Inherited from the parent experiment in this case
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

        if isinstance(port, Herald):
            if not self.are_modes_free([m], location):
                raise UnavailableModeException(m, "Another port overlaps")
            self._add_herald(m, port, location)
            return self

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
            if not self._find_and_remove_port_from_list(m, self._in_ports):
                raise UnavailableModeException(m, f"Port is not at location '{location.name}'")

        if location in (PortLocation.IN_OUT, PortLocation.OUTPUT):
            if not self._find_and_remove_port_from_list(m, self._out_ports):
                raise UnavailableModeException(m, f"Port is not at location '{location.name}'")
        return self

    def is_mode_connectible(self, mode: int) -> bool:
        if mode < 0:
            return False
        if mode >= self.circuit_size:
            return False
        return self._out_mode_type[mode] == ModeType.PHOTONIC

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

    @deprecated(version=0.12, reason="Add detectors as components")  # TODO (PCVL-935)
    def thresholded_output(self, value: bool):
        r"""
        Simulate threshold detectors on output states. All detections of more than one photon on any given mode is
        changed to 1.

        :param value: enables threshold detection when True, otherwise disables it.
        """
        self._thresholded_output = value
        self._detectors = [Detector.threshold() if self._thresholded_output else None] * len(self._detectors)

    @property
    @deprecated(version=0.12, reason="Use `detection_type` property instead")  # TODO (PCVL-935)
    def is_threshold(self) -> bool:
        return self._thresholded_output

    @property
    def detection_type(self) -> DetectionType:
        return get_detection_type(self._detectors)

    @property
    def heralds(self) -> dict[int, int]:
        return {port_range[0]: port.expected for port, port_range in self._out_ports.items() if isinstance(port, Herald)}

    @property
    def in_heralds(self) -> dict[int, int]:
        return {port_range[0]: port.expected for port, port_range in self._in_ports.items() if isinstance(port, Herald)}

    def check_input(self, input_state: BasicState):
        r"""Check if a basic state input matches with the current experiment configuration"""
        assert self.m_in, "A circuit has to be set before the input state"
        expected_input_length = self.m_in
        assert len(input_state) == expected_input_length, \
            f"Input length not compatible with circuit (expects {expected_input_length}, got {len(input_state)})"
        if input_state.has_polarization:
            get_logger().warn("Given input state has polarization, that will be ignored in the computation"
                              " (use with_polarized_input instead).")
        elif input_state.has_annotations:
            get_logger().warn("Given input state has annotations, that will be ignored in the computation."
                              " To use them, consider using a StateVector.")

    def _input_changed(self):
        for observer in self._input_changed_observers:
            observer()

    @dispatch(LogicalState)
    def with_input(self, input_state: LogicalState):
        input_state = get_basic_state_from_ports(list(self._in_ports.keys()), input_state)
        if self._min_detected_photons_filter is None:
            self._min_detected_photons_filter = input_state.n
        self.with_input(input_state)

    @dispatch(BasicState)
    def with_input(self, input_state: BasicState) -> None:
        self.check_input(input_state)
        input_list = [0] * self.circuit_size
        input_idx = 0
        # Build real input state (merging ancillas + expected input) and compute expected photon count
        for k in range(self.circuit_size):
            if k in self.in_heralds:
                input_list[k] = self.in_heralds[k]
            else:
                input_list[k] = input_state[input_idx]
                input_idx += 1

        self._input_state = BasicState(input_list)
        self._input_changed()

    @dispatch(StateVector)
    def with_input(self, sv: StateVector):
        r"""
        Setting directly state vector as input of a experiment, use SVDistribution input

        :param sv: the state vector
        """
        self.with_input(SVDistribution(sv))

    @dispatch(SVDistribution)
    def with_input(self, svd: SVDistribution):
        r"""
        Processor input can be set 100% manually via a state vector distribution, bypassing the source.

        :param svd: The input SVDistribution which won't be changed in any way by the source.
        Every state vector size has to be equal to `self.circuit_size`
        """
        assert self.m is not None, "A circuit has to be set before the input distribution"
        expected_photons = inf
        for sv in svd:
            for state in sv.keys():
                expected_photons = min(expected_photons, state.n)
                if state.m != self.circuit_size:
                    raise ValueError(
                        f'Input distribution contains states with a bad size ({state.m}), expected {self.circuit_size}')
        self._input_state = svd
        self._input_changed()

    def with_polarized_input(self, bs: BasicState):
        assert bs.has_polarization, "BasicState is not polarized, please use with_input instead"
        self._input_state = bs
        self._input_changed()

    def flatten(self, max_depth=None) -> list[tuple]:
        """
        List all the components in the experiment where recursive circuits have been flattened.

        :param max_depth: The maximum depth of recursion. The remaining sub-circuits at this depth are listed as a component.
        """
        return _flatten(self, max_depth=max_depth)


def _flatten(composite, starting_mode=0, max_depth=None) -> list[tuple]:
    component_list = []
    for m_range, comp in composite._components:
        if isinstance(comp, Circuit):
            if max_depth is None or max_depth > 0:
                sub_list = _flatten(comp, starting_mode=m_range[0],
                                    max_depth=max_depth - 1 if max_depth is not None else None)
                component_list += sub_list
            else:
                m_range = [m + starting_mode for m in m_range]
                component_list.append((m_range, comp))
        else:
            m_range = [m + starting_mode for m in m_range]
            component_list.append((m_range, comp))
    return component_list
