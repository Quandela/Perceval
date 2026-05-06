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

import dataclasses
from numbers import Number
from typing import Any

from perceval import Experiment, NoiseModel, PostSelect, BasicState


@dataclasses.dataclass
class ComputationDescriptor:
    experiment: Experiment
    max_shots: int | None
    max_samples: int | None
    parameters: dict[str, Any] | None = None


class ParameterIterator:
    # TODO: in the end, this is the class we want to send to the cloud that describes the computation
    # TODO: document this class in the code reference? I would say no as long as this is an internal class

    _ITERATOR_TYPE_CHECK: dict[str, type] = {'circuit_params': dict,
                                             'input_state': BasicState,
                                             'min_detected_photons': int,
                                             'max_samples': int,
                                             'max_shots': int,
                                             'noise': NoiseModel,
                                             'postselect': PostSelect}

    def __init__(self, experiment: Experiment, max_shots: int | None, max_samples: int | None):
        # Constant parameters
        self._experiment = experiment
        self._max_samples = max_samples
        self._max_shots = max_shots

        self._iterations: list[dict[str, Any]] = []

    @classmethod
    def from_payload(cls, payload: dict):
        """
        :param payload: A deserialized payload. Must contain an "experiment" field.
        :return: a ParameterIterator suited to simulate that payload
        """
        experiment = payload['experiment']
        n_shots = payload.get('max_shots')
        n_samples = payload.get('max_samples')
        self = cls(experiment, n_shots, n_samples)

        if "iterator" in payload:  # No check, we suppose the iterator has already been checked
            self._iterations = payload["iterator"]
        return self

    def to_payload(self):
        return {
            "experiment": self._experiment,
            "max_shots": self._max_shots,
            "max_samples": self._max_samples,
            "iterator": self._iterations
        }

    def check_sample_shot_iterator(self) -> bool:
        return all("max_samples" in it or "max_shots" in it for it in self._iterations)

    def input_available(self) -> bool:
        if self._experiment.input_state is not None:  # Default input will cover all cases
            return True
        elif len(self._iterations) == 0:  # ...else you need at least one iteration...
            return False
        else:
            for it in self._iterations:  # ...and all iterations must contain an input state
                if 'input_state' not in it:
                    return False
        return True

    @property
    def iterations(self) -> list[dict[str, Any]]:
        return self._iterations

    def _check_iteration(self, iter_params):
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
                    assert param_name in self._experiment.get_circuit_parameters(), \
                        f"Iteration: circuit parameter {param_name} does not exist in processor"
            elif key == 'input_state':
                assert val.m == self._experiment.m, \
                    f"Iteration: input state and processor size mismatch (processor size is {self._experiment.m})"
                self._experiment.check_input(iter_params['input_state'])

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

        # Iterator construction methods
        self._check_iteration(kwargs)
        self._iterations.append(kwargs)

    def clear_iterations(self):
        """
        Clear all prepared iterations.
        """
        self._iterations = []

    def __iter__(self):
        if len(self._iterations) == 0:  # no iterations
            yield ComputationDescriptor(self._experiment, self._max_shots, self._max_samples)
            return

        for it in self._iterations:
            yield self._apply_iteration(it)

    def __len__(self):
        return len(self._iterations)

    def __bool__(self):
        return bool(self._iterations)

    def _apply_iteration(self, it):
        # TODO: avoid copy if nothing in the iteration changes the Experiment
        computation = ComputationDescriptor(self._experiment.copy(), self._max_shots, self._max_samples)
        for key, val in it.items():
            try:
                self.__getattribute__(f"_set_{key}")(val, computation)
            except AttributeError:
                raise KeyError(f"Received unknown iteration key: {key}")
        computation.parameters = it
        return computation

    @staticmethod
    def _set_circuit_params(params: dict, computation: ComputationDescriptor):
        if params:
            circuit_params = computation.experiment.get_circuit_parameters()
            for name, value in params.items():
                if value is not None:
                    circuit_params[name].set_value(value)

    @staticmethod
    def _set_input_state(input_state: BasicState, computation: ComputationDescriptor):
        computation.experiment.with_input(input_state)

    @staticmethod
    def _set_min_detected_photons(count: int, computation: ComputationDescriptor):
        computation.experiment.min_detected_photons_filter(count)

    @staticmethod
    def _set_max_samples(val: int, computation: ComputationDescriptor):
        computation.max_samples = val

    @staticmethod
    def _set_max_shots(val: int, computation: ComputationDescriptor):
        computation.max_shots = val

    @staticmethod
    def _set_noise(noise: NoiseModel, computation: ComputationDescriptor):
        computation.experiment.noise = noise

    @staticmethod
    def _set_postselect(post_select: PostSelect, computation: ComputationDescriptor):
        computation.experiment.set_postselection(post_select)
