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
from typing import Callable, Dict

from perceval.components import ACircuit
from perceval.utils import BSDistribution, StateVector, SVDistribution


class ISimulator(ABC):
    @abstractmethod
    def set_circuit(self, circuit):
        pass

    @abstractmethod
    def probs(self, input_state) -> BSDistribution:
        pass

    @abstractmethod
    def probs_svd(self, svd: SVDistribution, progress_callback: Callable = None) -> Dict:
        pass

    @abstractmethod
    def evolve(self, input_state) -> StateVector:
        pass

    @abstractmethod
    def set_min_detected_photon_filter(self, value: int):
        pass


class ASimulatorDecorator(ISimulator, ABC):
    def __init__(self, simulator: ISimulator):
        self._simulator = simulator

    @abstractmethod
    def _prepare_input(self, input_state):
        pass

    @abstractmethod
    def _prepare_circuit(self, circuit) -> ACircuit:
        pass

    @abstractmethod
    def _postprocess_results(self, results):
        pass

    def set_circuit(self, circuit):
        self._simulator.set_circuit(self._prepare_circuit(circuit))

    def probs(self, input_state):
        results = self._simulator.probs(self._prepare_input(input_state))
        return self._postprocess_results(results)

    def probs_svd(self, svd: SVDistribution, progress_callback: Callable = None) -> Dict:
        probs = self._simulator.probs_svd(self._prepare_input(svd))
        probs['results'] = self._postprocess_results(probs['results'])
        return probs

    def evolve(self, input_state):
        results = self._simulator.evolve(self._prepare_input(input_state))
        return self._postprocess_results(results)

    def set_min_detected_photon_filter(self, value: int):
        self._simulator.set_min_detected_photon_filter(value)
