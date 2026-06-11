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
from typing import Any

from .parameter_iterator import ParameterIterator
from .command import Command
from .computation import Computation

from perceval.utils.constants import KEY_SHOTS_USED, KEY_MAX_SHOTS, KEY_MAX_SAMPLES, KEY_RESULTS_LIST, KEY_ITERATION
from perceval.components import Experiment


class ComputationIterator:
    """
    A computation consisting of several independent computations, where only a few parameters can change.

    This class modifies the results dict so that each individual result is inserted to a "results_list" field.
    """

    def __init__(self, base_computation: Computation):
        self.base_computation = base_computation
        # TODO: merge this with the ParameterIterator class instead of using it internally
        self._parameter_iterator = ParameterIterator(base_computation.experiment,
                                                     base_computation.parameters.get(KEY_MAX_SHOTS),
                                                     base_computation.parameters.get(KEY_MAX_SAMPLES))

    @property
    def command(self) -> Command:
        return self.base_computation.command

    @property
    def parameters(self) -> dict[str, Any]:
        return self.base_computation.parameters

    @property
    def experiment(self) -> Experiment:
        return self.base_computation.experiment

    @property
    def job_name(self) -> str:
        return self.base_computation.job_name

    @job_name.setter
    def job_name(self, value: str):
        self.base_computation.job_name = value

    @property
    def job_group_name(self) -> str | None:
        return self.base_computation.job_group_name

    @job_group_name.setter
    def job_group_name(self, value: str):
        self.base_computation.job_group_name = value

    def __iter__(self):
        if len(self._parameter_iterator) == 0:
            yield self.base_computation

        for iteration in self._parameter_iterator:
            computation = Computation(self.base_computation.command, iteration.experiment)
            if iteration.max_samples is not None:
                computation.add_params(max_samples=iteration.max_samples)
            if iteration.max_shots is not None:
                computation.add_params(max_shots = iteration.max_shots)
            yield computation

    def __len__(self):
        return len(self._parameter_iterator)

    def __bool__(self):
        return bool(self._parameter_iterator)

    def clear_iterations(self):
        """
        Clear all prepared iterations.
        """
        self._parameter_iterator.clear_iterations()

    def add_iteration(self, **kwargs):
        """
        Add a single iteration to future jobs.

        :param kwargs: List of accepted keywords:

           - circuit_params: dict containing pairs (parameter_name: str - value : number)
           - input_state: BasicState
           - min_detected_photons: int
           - max_samples: int
           - max_shots: int
           - noise: NoiseModel
           - postselect: PostSelect
        """
        self._parameter_iterator.add_iteration(**kwargs)

    @staticmethod
    def validate() -> bool:
        return True  # Already done by the ParameterIterator

    # Follows the Mitigation interface for automatic parsing
    def extend_computation(self, *args, **kwargs) -> list[Computation]:
        return [comp for comp in self]

    def parse_results(self, computation: Computation, results: list, noise) -> dict:
        """
        Parses the results obtained from an iterator obtained through extend_computation().
        :param computation: The computation asked by the upper layer
        :param results: The results for the list of computations obtained through extend_computation()
        :param noise: The Computer noise with which the results were obtained
        :return: A dictionary containing the results of the computation as a list in the "results_list" field.
        """
        if len(self._parameter_iterator) == 0:
            return results[0]

        res = {KEY_RESULTS_LIST: results}
        n_shots = None
        for result, iteration in zip(results, self._parameter_iterator):
            result[KEY_ITERATION] = iteration.parameters
            if KEY_SHOTS_USED in result:
                n_shots = result[KEY_SHOTS_USED] if n_shots is None else n_shots + result[KEY_SHOTS_USED]

        return res
