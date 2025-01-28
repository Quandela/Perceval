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

from perceval.components import ACircuit, IDetector
from perceval.utils import BSDistribution, StateVector, SVDistribution, PostSelect, post_select_distribution, \
    post_select_statevector, filter_distribution_photon_count
from perceval.utils.logging import deprecated, get_logger


class ISimulator(ABC):

    def __init__(self):
        self._silent = False
        self._postselect: PostSelect = PostSelect()
        self._heralds: dict = {}
        self._min_detected_photons_filter: int = 0

    def set_silent(self, silent: bool):
        self._silent = silent

    @abstractmethod
    def set_circuit(self, circuit, m = None):
        pass

    @abstractmethod
    def probs(self, input_state) -> BSDistribution:
        pass

    @abstractmethod
    def probs_svd(self,
                  svd: SVDistribution,
                  detectors: list[IDetector] = None,
                  progress_callback: callable = None) -> dict:
        pass

    @abstractmethod
    def evolve(self, input_state) -> StateVector:
        pass

    @deprecated(version="0.11.1", reason="Use set_min_detected_photons_filter instead")
    def set_min_detected_photon_filter(self, value: int):
        """
        Set a minimum number of detected photons in the output distribution

        :param value: The minimum photon count
        """
        self.set_min_detected_photons_filter(value)

    def set_min_detected_photons_filter(self, value: int):
        """
        Set a minimum number of detected photons in the output distribution

        :param value: The minimum photon count
        """
        self._min_detected_photons_filter = value

    def set_precision(self, precision: float):
        pass

    def set_selection(self,
                      min_detected_photons_filter: int = None,
                      postselect: PostSelect = None,
                      heralds: dict = None,
                      min_detected_photon_filter: int = None):  # TODO: remove for PCVL-786
        if min_detected_photon_filter is not None:  # TODO: remove for PCVL-786
            get_logger().warn(
                'DeprecationWarning: Call with deprecated argument "min_detected_photon_filter", please use "min_detected_photons_filter" instead')
            min_detected_photons_filter = min_detected_photon_filter
        if min_detected_photons_filter is not None:
            self.set_min_detected_photons_filter(min_detected_photons_filter)
        if postselect is not None:
            self._postselect = postselect
        if heralds is not None:
            self._heralds = heralds


class ASimulatorDecorator(ISimulator, ABC):
    def __init__(self, simulator: ISimulator):
        super().__init__()
        self._simulator: ISimulator = simulator

    @abstractmethod
    def _prepare_input(self, input_state):
        pass

    @abstractmethod
    def _prepare_circuit(self, circuit, m = None) -> ACircuit:
        pass

    @abstractmethod
    def _postprocess_bsd_impl(self, bsd: BSDistribution) -> BSDistribution:
        pass

    @abstractmethod
    def _postprocess_sv_impl(self, sv: StateVector) -> StateVector:
        pass

    def _postprocess_bsd(self, results: BSDistribution):
        results = self._postprocess_bsd_impl(results)
        physical_perf = 1
        if self._min_detected_photons_filter:
            results, physical_perf = filter_distribution_photon_count(results, self._min_detected_photons_filter)
        logical_perf = 1
        if self._postselect is not None or self._heralds is not None:
            # Only at last layer since postselect and heralds are not transmitted
            results, logical_perf = post_select_distribution(results, self._postselect, self._heralds)
        return results, logical_perf, physical_perf

    def _postprocess_sv(self, sv: StateVector) -> StateVector:
        sv = self._postprocess_sv_impl(sv)
        if self._postselect is not None or self._heralds is not None:
            sv, _ = post_select_statevector(sv, self._postselect, self._heralds)
        return sv

    def set_circuit(self, circuit, m = None):
        self._simulator.set_circuit(self._prepare_circuit(circuit, m))

    def probs(self, input_state) -> BSDistribution:
        results = self._simulator.probs(self._prepare_input(input_state))
        results = self._postprocess_bsd(results)[0]
        return results

    def probs_svd(self, svd: SVDistribution, detectors=None, progress_callback: callable = None) -> dict:
        probs = self._simulator.probs_svd(self._prepare_input(svd),
                                          progress_callback=progress_callback)
        probs['results'], logical_perf_coeff, physical_perf_coeff = self._postprocess_bsd(probs['results'])
        probs['physical_perf'] *= physical_perf_coeff
        probs['logical_perf'] *= logical_perf_coeff
        return probs

    def evolve(self, input_state) -> StateVector:
        results = self._simulator.evolve(self._prepare_input(input_state))
        return self._postprocess_sv(results)

    def set_min_detected_photons_filter(self, value: int):
        super().set_min_detected_photons_filter(value)
        self._simulator.set_min_detected_photons_filter(value)

    def set_precision(self, precision: float):
        self._simulator.set_precision(precision)
