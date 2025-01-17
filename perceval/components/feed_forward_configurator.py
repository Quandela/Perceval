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

from .unitary_components import Unitary
from .abstract_component import AComponent
from .linear_circuit import ACircuit
from perceval.utils import BasicState, Matrix


class AFFConfigurator(AComponent, ABC):
    DEFAULT_NAME = "FFC"

    """
    Abstract feed-forward configurator.

    :param m: The number of classical modes that are detected (after a detector)
    :param offset: The distance between the configurator and the first mode of the implemented circuits.
        For positive values, it is the number of empty modes between the configurator and the configured circuit below.
        For negative values, it is the same but the circuit is located above the configurator
        (the number of empty modes is abs(`offset`) - 1,
        so an offset of -1 means that there is no empty modes between the configurator and the circuit).
        All circuits are considered to have the size of the biggest possible circuit in this configurator.
    :param default_circuit: The circuit to be used if the measured state does not befall into one of the declared cases
    """

    def __init__(self, m: int, offset: int, default_circuit: ACircuit, name: str = None):
        super().__init__(m, name)
        self._offset = offset
        self.default_circuit = default_circuit
        self._max_circuit_size = default_circuit.m
        self._blocked_circuit_size = False

    def block_circuit_size(self):
        """Call this to prevent adding circuits bigger than the current maximum size"""
        self._blocked_circuit_size = True

    @abstractmethod
    def configure(self, measured_state: BasicState) -> ACircuit:
        """
        Gives the circuit or processor that must be configured given the measured state

        :param measured_state: The state of size self.m that corresponds to the measurements
         of the modes on which the configurator is located.
        :return: The processor or circuit that must be set
        """
        pass

    @abstractmethod
    def circuit_template(self) -> ACircuit:
        """
        Return a fitting representation of the controlled circuit or processor.
        """

    def config_modes(self, self_modes: tuple[int, ...]) -> tuple[int, ...]:
        """
        Gives the list of modes on which to place the circuit

        :param self_modes: The tuple containing the modes on which the configurator is located, in crescent order
        """
        assert len(self_modes) == self.m, "Incorrect number of modes"
        if self._offset >= 0:
            first_mode = self_modes[-1] + 1 + self._offset
            return tuple(range(first_mode, first_mode + self._max_circuit_size))

        last_mode = self_modes[0] + self._offset
        return tuple(range(last_mode - self._max_circuit_size + 1, last_mode + 1))

    @property
    def circuit_offset(self):
        return self._offset

    @circuit_offset.setter
    def circuit_offset(self, offset: int):
        assert isinstance(offset, int), f"A feed-forward configurator offset must be an integer (received {offset})"
        self._offset = offset

    def copy(self, subs=None) -> AFFConfigurator:
        return copy.copy(self)


class FFCircuitProvider(AFFConfigurator):
    DEFAULT_NAME = "FFC"

    """
    For any measurement, FFCircuitProvider will return a circuit or a processor, picked from known mapping of configurations.
    Each configuration links a measurement to a circuit or processor.
    If a measurement is received and was not set in the mapping, a mandatory default circuit or processor is returned.

    :param m: The number of classical modes that are detected (after a detector)
    :param offset: The distance between the configurator and the first mode of the implemented circuits.
        For positive values, it is the number of empty modes between the configurator and the configured circuit below.
        For negative values, it is the same but the circuit is located above the configurator
        (the number of empty modes is abs(`offset`) - 1,
        so an offset of -1 means that there is no empty modes between the configurator and the circuit).
        All circuits are considered to have the size of the biggest possible circuit in this configurator.
    :param default_circuit: The circuit to be used if the measured state does not befall into one of the declared cases
    """

    def __init__(self, m: int, offset: int, default_circuit: ACircuit, name: str = None):
        assert not isinstance(default_circuit, AFFConfigurator), \
            "Can't add directly a Feed-forward configurator to a configurator (use a Processor)"
        super().__init__(m, offset, default_circuit, name)
        self._map: dict[BasicState, ACircuit] = {}

    def reset_map(self):
        self._max_circuit_size = self.default_circuit.m
        self._map = {}

    @property
    def circuit_map(self):
        return self._map

    @circuit_map.setter
    def circuit_map(self, circuit_map: dict[BasicState, ACircuit]):
        self.reset_map()
        for state, circ in circuit_map.items():
            self.add_configuration(state, circ)

    def add_configuration(self, state, circuit: ACircuit) -> FFCircuitProvider:
        state = BasicState(state)
        assert state.m == self.m, f"Incorrect number of modes for state {state} (expected {self.m})"
        assert not isinstance(circuit, AFFConfigurator), \
            "Can't add directly a Feed-forward configurator to a configurator (use a Processor)"
        if not self._blocked_circuit_size:
            self._max_circuit_size = max(self._max_circuit_size, circuit.m)
        else:
            if circuit.m != self._max_circuit_size:
                raise RuntimeError(f"Circuit size mismatch (got {circuit.m}, expected {self._max_circuit_size} modes)")
        self._map[state] = circuit

        return self

    def configure(self, measured_state: BasicState) -> ACircuit:
        return self.circuit_map.get(measured_state, self.default_circuit)

    def circuit_template(self) -> ACircuit:
        return Unitary(Matrix.eye(self.default_circuit.m), f"U({self.name})")


class FFConfigurator(AFFConfigurator):
    DEFAULT_NAME = "FFC"

    """
    This class relies on a mapping between detections and a mapping of variable names and numerical values, controlling
    a circuit template.

    :param m: The number of classical modes that are detected (after a detector)
    :param offset: The distance between the configurator and the first mode of the implemented circuits.
        For positive values, it is the number of empty modes between the configurator and the configured circuit below.
        For negative values, it is the same but the circuit is located above the configurator
        (the number of empty modes is abs(`offset`) - 1,
        so an offset of -1 means that there is no empty modes between the configurator and the circuit).
        All circuits are considered to have the size of the biggest possible circuit in this configurator.
    :param controlled_circuit: A circuit containing symbolic parameters whose value will be changed depending on the 
        measured state.
    :param default_config: A dictionary mapping the parameters of the circuit and their value to use in case a measured 
        state does not befall into one of the declared cases.
    """

    def __init__(self,
                 m: int,
                 offset: int,
                 controlled_circuit: ACircuit,
                 default_config: dict[str, float],
                 name: str = None):
        if not isinstance(controlled_circuit, ACircuit):
            raise TypeError(f"controlled_circuit must be of type ACircuit")
        self._controlled = controlled_circuit
        self._linked_vars = self._controlled.vars
        self._configs = {}
        self._check_configuration(default_config)
        default_circuit = controlled_circuit.copy()
        default_circuit.assign(default_config)
        super().__init__(m, offset, default_circuit, name)

    def _check_configuration(self, config: dict[str, float]):
        if len(config) != len(self._linked_vars):
            raise ValueError(
                f"Wrong parameter count in the configuration ({len(config)}, expected {len(self._linked_vars)})")
        for param_name in config:
            if param_name not in self._linked_vars:
                raise NameError(f"Parameter {param_name} does not exist in the controlled circuit")

    def add_configuration(self, detections: BasicState | tuple[int, ...], config: dict[str, float]) -> FFConfigurator:
        detections = BasicState(detections)
        if detections.m != self.m:
            raise ValueError(f"Wrong size for detections; got {len(detections)}, expected the number of modes plugged-in, i.e. {self.m}")
        self._check_configuration(config)
        self._configs[detections] = config
        return self

    def configure(self, measured_state: BasicState) -> ACircuit:
        if measured_state not in self._configs:
            return self.default_circuit

        circuit = self._controlled.copy()
        circuit.assign(self._configs[measured_state])
        return circuit

    def circuit_template(self) -> ACircuit:
        return self._controlled
