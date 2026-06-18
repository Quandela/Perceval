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
from copy import deepcopy
from numbers import Number
from typing import Any, Callable

from .command import Command
from .computation import Computation

from perceval.utils import BasicState, PostSelect
from perceval.utils.constants import KEY_SHOTS_USED, KEY_RESULTS_LIST, KEY_ITERATION
from perceval.components import Experiment
from perceval.serialization import register_to_serialization


class ComputationIterator:
    """
    A computation consisting of several independent computations, where only a few parameters can change.

    This class modifies the results dict so that each individual result is inserted to a "results_list" field.
    """

    _ITERATOR_TYPE_CHECK: dict[str, type] = {'circuit_params': dict,
                                             'input_state': BasicState,
                                             'min_detected_photons': int,
                                             'max_samples': int,
                                             'max_shots': int,
                                             'postselect': PostSelect,
                                             'compilation_seed': int,}

    def __init__(self, base_computation: Computation):
        self.base_computation = base_computation
        self._iterations: list[dict[str, Any]] = []

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

    @property
    def iterations(self) -> list[dict[str, Any]]:
        return self._iterations

    def _check_iteration(self, iter_params: dict):
        assert isinstance(iter_params, dict), "Iteration parameters must be a valid dictionary"
        for key, val in iter_params.items():
            if key in self._ITERATOR_TYPE_CHECK:
                correct_type = self._ITERATOR_TYPE_CHECK[key]
                assert isinstance(val, correct_type), \
                    (f"Iteration: unexpected type for {key}, expected {correct_type.__name__},"
                     f" received {type(val).__name__}")
            else:
                raise NotImplementedError(f"Iteration: received unknown key {key}")

            # Further checks
            if key == 'circuit_params':
                for param_name, param_value in val.items():
                    assert isinstance(param_value, Number), \
                        f"Iteration: circuit parameters have to be numerical values (got {param_value})"
                    assert param_name in self.experiment.get_circuit_parameters(), \
                        f"Iteration: circuit parameter {param_name} does not exist in processor"
            elif key == 'input_state':
                assert val.m == self.experiment.m, \
                    f"Iteration: input state and processor size mismatch (processor size is {self.experiment.m})"
                self.experiment.check_input(iter_params['input_state'])

    def add_iteration(self, **kwargs):
        """
        Add a single iteration to future jobs.

        :param kwargs: List of accepted keywords:

           - circuit_params: dict containing pairs (parameter_name: str - value : number)
           - input_state: BasicState
           - min_detected_photons: int
           - max_samples: int
           - max_shots: int
           - postselect: PostSelect
           - compilation_seed: int
        """

        # Iterator construction methods
        self._check_iteration(kwargs)
        self._iterations.append(kwargs)

    def clear_iterations(self):
        """
        Clear all prepared iterations.
        """
        self._iterations = []

    def __len__(self):
        return len(self._iterations)

    def __bool__(self):
        return bool(self._iterations)

    def __iter__(self):
        if len(self._iterations) == 0:
            yield self.base_computation

        for iteration in self._iterations:
            yield self._apply_iteration(iteration)

    def _apply_iteration(self, it: dict):
        computation = deepcopy(self.base_computation)
        for key, val in it.items():
            try:
                self.__getattribute__(f"_set_{key}")(val, computation)
            except AttributeError:
                raise KeyError(f"Received unknown iteration key: {key}")
        return computation

    def validate(self) -> bool:
        # Already done by the _check_iteration at construction for other computations
        return self.base_computation.validate()

    def make_inserter(self, out: dict) -> Callable[[dict], None]:
        """
        :param out: The place where to store the results of the computation
        :return: A callable that can be used to add results to :code:`out`
        """
        out[KEY_RESULTS_LIST] = []

        def inserter(res: dict):
            i = len(out[KEY_RESULTS_LIST])
            res[KEY_ITERATION] = self._iterations[i]
            if KEY_SHOTS_USED in res:
                out[KEY_SHOTS_USED] = out[KEY_SHOTS_USED] + res[KEY_SHOTS_USED] if KEY_SHOTS_USED in out else res[KEY_SHOTS_USED]
            out[KEY_RESULTS_LIST].append(res)

        return inserter

    @staticmethod
    def _set_circuit_params(params: dict, computation: Computation):
        if params:
            circuit_params = computation.experiment.get_circuit_parameters()
            for name, value in params.items():
                if value is not None:
                    circuit_params[name].set_value(value)

    @staticmethod
    def _set_input_state(input_state: BasicState, computation: Computation):
        computation.experiment.with_input(input_state)

    @staticmethod
    def _set_min_detected_photons(count: int, computation: Computation):
        computation.experiment.min_detected_photons_filter(count)

    @staticmethod
    def _set_max_samples(val: int, computation: Computation):
        computation.add_params(max_samples = val)

    @staticmethod
    def _set_max_shots(val: int, computation: Computation):
        computation.add_params(max_shots = val)

    @staticmethod
    def _set_postselect(post_select: PostSelect, computation: Computation):
        computation.experiment.set_postselection(post_select)

    @staticmethod
    def _set_compilation_seed(compilation_seed: int, computation: Computation):
        computation.add_params(compilation_seed = compilation_seed)

register_to_serialization(ComputationIterator, default_compress=True)
