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

from abc import abstractmethod, ABC

from exqalibur import BSCount, BSSamples

from ..computation import Computation

from perceval.utils import NoiseModel, ConversionHelper, apply_min_photons, apply_post_select, BSDistribution
from perceval.utils.constants import KEY_RESULTS, KEY_GLOBAL_PERF, KEY_PHYSICAL_PERF, KEY_LOGICAL_PERF
from perceval.components import Experiment


class AbstractMitigation(ABC):

    APPLY_MIN_PHOTONS = True  # By default, avoid any accident at the cost of performance
    APPLY_LOGICAL_SELECTION = True

    @abstractmethod
    def extend_computation(self, computation: Computation, noise: NoiseModel) -> list[Computation]:
        """
        :param computation: The computation asked by the upper layer
        :param noise: The Computer noise
        :return: a list of all computations to execute to apply the mitigation
        """
        pass

    @abstractmethod
    def _parse_results(self, computation: Computation, results: list[dict], noise: NoiseModel) -> dict:
        """
        Parses the results obtained from an iterator obtained through extend_computation().
        :param results: The results for the list of computations obtained through extend_computation()
        :param noise: The Computer noise with which the results were obtained
        :return: A dict with the fields "results", "global_perf", "nb_shots_used"
        """
        pass

    def parse_results(self, computation: Computation, results: list[dict], noise: NoiseModel) -> dict:
        """
        Parses the results obtained from an iterator obtained through extend_computation().
        :param computation: The computation asked by the upper layer
        :param results: The results for the list of computations obtained through extend_computation()
        :param noise: The Computer noise with which the results were obtained
        :return: The mitigated result, matching the expectations of computation
        """
        result = self._parse_results(computation, results, noise)

        res, physical_perf, logical_perf = self._apply_filtering(computation.experiment, result[KEY_RESULTS])

        # TODO: find a way to transmit the correct number of states between layers
        #       We should not use computation.parameters
        res = ConversionHelper.convert_to(computation.command.name, res, **computation.parameters)
        result[KEY_RESULTS] = res

        result[KEY_GLOBAL_PERF] *= physical_perf * logical_perf
        if KEY_PHYSICAL_PERF in result:
            result[KEY_PHYSICAL_PERF] *= physical_perf
        if KEY_LOGICAL_PERF in result:
            result[KEY_LOGICAL_PERF] *= logical_perf

        return result

    def _apply_filtering(self, experiment: Experiment, result: BSDistribution | BSCount | BSSamples) -> tuple[BSDistribution | BSCount | BSSamples, float, float]:
        if self.APPLY_MIN_PHOTONS:
            min_photons = experiment.min_photons_filter or 0
            result, physical_perf = apply_min_photons(result, min_photons)
        else:
            physical_perf = 1.

        if self.APPLY_LOGICAL_SELECTION:
            result, logical_perf = apply_post_select(result, experiment.post_select_fn, experiment.heralds, False)
        else:
            logical_perf = 1.

        return result, physical_perf, logical_perf
