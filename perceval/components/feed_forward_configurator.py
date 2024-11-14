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
from abc import ABC, abstractmethod

from .abstract_component import AComponent
from .processor import Processor
from .linear_circuit import ACircuit
from perceval.utils.statevector import BasicState


class AFFConfigurator(AComponent, ABC):
    DEFAULT_NAME = "FFC"

    def __init__(self, m: int, offset: int, default_circuit: ACircuit | Processor):
        """
        A feed-forward configurator.

        :param m: The number of classical modes that are detected (after a detector)
        :param offset: The distance between the configurator and the first mode of the implemented circuits.
         For positive values, it is the number of empty modes between the configurator and the configured circuit below.
         For negative values, it is the same but the circuit is located above the configurator
         (the number of empty modes is abs(`offset`) - 1,
         so an offset of -1 means that there is no empty modes between the configurator and the circuit).
         All circuits are considered to have the size of the biggest possible circuit in this configurator.
        :param default_circuit: The circuit to be used if the measured state does not befall into one of the declared cases
        """
        super().__init__(m)
        self._offset = offset
        self.default_circuit = default_circuit
        self._max_circuit_size = default_circuit.m
        self._blocked_circuit_size = False

    def block_circuit_size(self):  # TODO: turn this to True when adding to a processor
        """Call this to prevent adding circuits bigger than the current maximum size"""
        self._blocked_circuit_size = True

    @abstractmethod
    def configure(self, measured_state: BasicState) -> Processor | ACircuit:
        """
        Gives the circuit or processor that must be configured given the measured state

        :param measured_state: The state of size self.m that corresponds to the measurements
         of the modes on which the configurator is located.
        :return: The processor or circuit that must be set
        """
        pass

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


class CircuitMapFFConfig(AFFConfigurator):

    def __init__(self, m: int, offset: int, default_circuit: ACircuit | Processor):
        super().__init__(m, offset, default_circuit)
        self._map: dict[BasicState, ACircuit | Processor] = {}

    def reset_map(self):
        self._max_circuit_size = self.default_circuit.m
        self._map = {}

    @property
    def circuit_map(self):
        return self._map

    @circuit_map.setter
    def circuit_map(self, circuit_map: dict[BasicState, ACircuit | Processor]):
        self.reset_map()
        for state, circ in circuit_map.items():
            self.add_configuration(state, circ)

    def add_configuration(self, state, circuit: ACircuit | Processor) -> CircuitMapFFConfig:
        state = BasicState(state)
        assert state.m == self.m, f"Incorrect number of modes for state {state} (expected {self.m})"
        if not self._blocked_circuit_size:
            self._max_circuit_size = max(self._max_circuit_size, circuit.m)
        else:
            assert circuit.m <= self._max_circuit_size, \
                f"Circuit is too big for this configurator (got {circuit.m} modes, expected at most {self._max_circuit_size} modes"
        self._map[state] = circuit

        return self

    def configure(self, measured_state: BasicState) -> Processor | ACircuit:
        return self.circuit_map.get(measured_state, self.default_circuit)
