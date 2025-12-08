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
from __future__ import annotations  # Python 3.11 : Replace using Self typing

from abc import ABC, abstractmethod
from enum import Enum

from perceval.utils import BasicState, FockState, Parameter, PostSelect, LogicalState, NoiseModel, SVDistribution, StateVector
from perceval.components.abstract_component import AComponent
from perceval.components.detector import DetectionType
from perceval.components.experiment import Experiment
from perceval.components.linear_circuit import Circuit, ACircuit
from perceval.components.port import PortLocation, APort


class ProcessorType(Enum):
    SIMULATOR = 1
    PHYSICAL = 2


class AProcessor(ABC):
    def __init__(self, experiment = None):
        self.experiment = experiment or Experiment()
        self._parameters: dict[str, any] = {}

    @property
    def experiment(self) -> Experiment:
        return self._experiment

    @experiment.setter
    def experiment(self, experiment: Experiment):
        self._experiment = experiment
        experiment.add_observers(self._circuit_change_observer,
                                 self._noise_changed_observer,
                                 self._input_changed_observer)

    @abstractmethod
    def _circuit_change_observer(self, new_component = None):
        pass

    @abstractmethod
    def _noise_changed_observer(self):
        pass

    def _input_changed_observer(self):
        pass

    @property
    def name(self) -> str:
        return self.experiment.name

    @name.setter
    def name(self, name: str):
        self.experiment.name = name

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
        self.experiment.clear_input_and_circuit(new_m)

    @property
    def _min_detected_photons_filter(self):
        return self.experiment.min_photons_filter

    def min_detected_photons_filter(self, n: int):
        r"""
        Sets-up a state post-selection on the number of detected photons. With threshold detectors, this will
        actually filter on "click" count.

        :param n: Minimum expected photons. Does not take heralded modes into account.

        This post-selection has an impact on the output physical performance
        """
        self.experiment.min_detected_photons_filter(n)

    @property
    def input_state(self):
        return self.experiment.input_state

    @property
    def noise(self):
        return self.experiment.noise

    @noise.setter
    def noise(self, nm: NoiseModel):
        self.experiment.noise = nm

    @property
    @abstractmethod
    def available_commands(self) -> list[str]:
        pass

    def remove_heralded_modes(self, s: FockState) -> FockState:
        if self.heralds:
            s = s.remove_modes(list(self.heralds.keys()))
        return s

    @property
    def post_select_fn(self):
        return self.experiment.post_select_fn

    def set_postselection(self, postselect: PostSelect):
        r"""
        Set a logical post-selection function. Along with the heralded modes, this function has an impact
        on the logical performance of the processor

        :param postselect: Sets a post-selection function. Its signature must be `func(s: BasicState) -> bool`.
            If None is passed as parameter, removes the previously defined post-selection function.
        """
        self.experiment.set_postselection(postselect)

    def clear_postselection(self):
        self.experiment.clear_postselection()

    @abstractmethod
    def compute_physical_logical_perf(self, value: bool):
        pass

    def _state_selected(self, state: BasicState) -> bool:
        """
        Computes if the state is selected given heralds and post selection function
        """
        # TODO: see what to do with this method
        for m, v in self.heralds.items():
            if state[m] != v:
                return False
        if self.experiment.post_select_fn is not None:
            return self.experiment.post_select_fn(state)
        return True

    def set_circuit(self, circuit: ACircuit) -> AProcessor:
        r"""
        Removes all components and replace them by the given circuit.

        :param circuit: The circuit to start the processor with
        :return: Self to allow direct chain this with .add()
        """
        self.experiment.set_circuit(circuit)
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
        self.experiment.add(mode_mapping, component, keep_port)
        return self

    @property
    def detectors(self):
        return self.experiment.detectors

    def add_herald(self, mode: int, expected: int, name: str = None, location: PortLocation = PortLocation.IN_OUT) -> AProcessor:
        r"""
        Add a heralded mode

        :param mode: Mode index of the herald
        :param expected: number of expected photon as input and/or output on the given mode
        :param name: Herald port name. If none is passed, the name is auto-generated
        :param location: Port location of the herald (input, output or both)
        """
        self.experiment.add_herald(mode, expected, name, location)
        return self

    @property
    def components(self):
        return self.experiment.components

    @property
    def m(self) -> int:
        """
        :return: The number of modes that are free (non-heralded) on the output
        """
        return self.experiment.m

    @property
    def m_in(self) -> int:
        """
        :return: The number of modes that are free (non-heralded) on the input,
                that must be specified when using self.with_input()
        """
        return self.experiment.m_in

    @m.setter
    def m(self, value: int):
        self.experiment.m = value

    @property
    def circuit_size(self) -> int:
        r"""
        :return: Total size of the enclosed circuit (i.e. self.m + heralded mode count)
        """
        return self.experiment.circuit_size

    def linear_circuit(self, flatten: bool = False) -> Circuit:
        """
        Creates a linear circuit from internal components, if all internal components are unitary.
        :param flatten: if True, the component recursive hierarchy is discarded, making the output circuit "flat".
        """
        return self.experiment.unitary_circuit(flatten=flatten)

    def non_unitary_circuit(self, flatten: bool = False) -> list[tuple[tuple, AComponent]]:
        return self.experiment.non_unitary_circuit(flatten=flatten)

    def get_circuit_parameters(self) -> dict[str, Parameter]:
        return self.experiment.get_circuit_parameters()

    @property
    def out_port_names(self):
        r"""
        :return: A list of the output port names. Names are repeated for ports connected to more than one mode
        """
        return self.experiment.out_port_names

    @property
    def in_port_names(self):
        r"""
        :return: A list of the input port names. Names are repeated for ports connected to more than one mode
        """
        return self.experiment.in_port_names

    def add_port(self, m, port: APort, location: PortLocation = PortLocation.IN_OUT):
        self.experiment.add_port(m, port, location)
        return self

    def remove_port(self, m, location: PortLocation = PortLocation.IN_OUT):
        self.experiment.remove_port(m, location)
        return self

    def get_input_port(self, mode):
        return self.experiment.get_input_port(mode)

    def get_output_port(self, mode):
        return self.experiment.get_output_port(mode)

    @property
    def detection_type(self) -> DetectionType:
        return self.experiment.detection_type

    @property
    def heralds(self):
        """
        :return: A dictionary {mode: expected_count} describing the heralds on the output
        """
        return self.experiment.heralds

    @property
    def in_heralds(self):
        """
        :return: A dictionary {mode: expected_count} describing the heralds on the input
        """
        return self.experiment.in_heralds

    def check_input(self, input_state: FockState):
        r"""Check if a Fock state input matches with the current processor configuration"""
        self.experiment.check_input(input_state)

    def check_min_detected_photons_filter(self):
        if self._min_detected_photons_filter is None:
            if (not self.is_remote and self._source is not None and self._source.is_perfect()
                    and isinstance(self.input_state, BasicState)):
                # Automatically set the min_detected_photons_filter for perfect sources of local processors if not set
                self.min_detected_photons_filter(self.input_state.n - sum(self.heralds.values()))
            else:
                raise ValueError("The value of min_detected_photons is not set."
                                 " Use the method processor.min_detected_photons_filter(value).")

    def with_input(self, input_state: BasicState | LogicalState | StateVector | SVDistribution):
        """
        Simulates plugging the photonic source on certain modes and turning it on.
        Computes the input probability distribution

        :param input_state: Expected input BasicState, StateVector or SVDistribution of length `self.m`
         (heralded modes are managed automatically),
         or a LogicalState with length 'number of non-herald ports or port-free modes'.

        The properties of the source will alter the input state for BasicState and LogicalState inputs.
        A perfect source always delivers the expected state as an input. Imperfect ones won't.
        """
        self.experiment.with_input(input_state)

    def flatten(self, max_depth = None) -> list[tuple]:
        """
        List all the components in the processor where recursive circuits have been flattened.

        :param max_depth: The maximum depth of recursion. The remaining sub-circuits at this depth are listed as a component.
        """
        return self.experiment.flatten(max_depth)
