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
from typing import Callable, List, Dict
from numbers import Number

from .abstract_algorithm import AAlgorithm
from perceval.utils import samples_to_sample_count, samples_to_probs, sample_count_to_samples,\
                           sample_count_to_probs, probs_to_samples, probs_to_sample_count
from perceval.components.abstract_processor import AProcessor
from perceval.runtime import Job, RemoteJob, LocalJob
from perceval.utils import BasicState


class Sampler(AAlgorithm):
    """
    Base algorithm able to retrieve some sampling results via 3 methods
    - samples(max_samples) : returns a list of sampled states
    - sample_count(max_samples) : return a table (output state) : (sampled count)
                            the sum of 'sampled counts' can slightly differ from requested 'count'
    - probs() : returns a probability distribution of output states

    The form of the output for all 3 sampling methods is a dictionary containing a 'results' key and several performance
    values.
    """
    PROBS_SIMU_SAMPLE_COUNT = 10000  # Arbitrary value
    # Absolute maximum of samples for local sampling simulation (should be thresholded by a max_shot value)
    SAMPLES_MAX_COUNT = int(1e8)
    _METHOD_MAPPING = {
        'probs': {'sample_count': sample_count_to_probs, 'samples': samples_to_probs},
        'sample_count': {'probs': probs_to_sample_count, 'samples': samples_to_sample_count},
        'samples': {'probs': probs_to_samples, 'sample_count': sample_count_to_samples}
    }

    def __init__(self, processor: AProcessor, **kwargs):
        super().__init__(processor, **kwargs)
        self._iterator = []

    def _get_primitive_converter(self, method: str):
        available_primitives = self._processor.available_commands
        if method in available_primitives:
            return method, None
        if method in self._METHOD_MAPPING:
            pmap = self._METHOD_MAPPING[method]
            for k, converter in pmap.items():
                if k in available_primitives:
                    return k, converter
        return None, None

    def _input_available(self) -> bool:
        if self._processor.input_state is not None:  # Default input will cover all cases
            return True
        elif len(self._iterator) == 0:  # ...else you need at least one iteration...
            return False
        else:
            for it in self._iterator:  # ...and all iterations must contain an input state
                if 'input_state' not in it:
                    return False
        return True

    # Job creation methods
    def _create_job(self, method: str):
        assert self._input_available(), "Missing input state"
        primitive, converter = self._get_primitive_converter(method)
        if primitive is None:
            raise RuntimeError(
                f"cannot find a compatible primitive to execute {method} in {self._processor.available_commands}")
        method_is_probs = (method.find('sample') == -1)
        primitive_is_probs = (primitive.find('sample') == -1)

        delta_parameters = {"command": {}, "mapping": {}}
        # adapt the parameters list
        command_param_names = [] if primitive_is_probs else ['max_samples']
        if not method_is_probs and primitive_is_probs:
            delta_parameters["mapping"]['max_samples'] = None  # Is to be filled be job._handle_params
            delta_parameters["mapping"]['max_shots'] = self._max_shots
        elif method_is_probs and not primitive_is_probs:
            delta_parameters["command"]['max_samples'] = self.PROBS_SIMU_SAMPLE_COUNT
        elif not method_is_probs and not primitive_is_probs:
            delta_parameters["command"]['max_samples'] = None  # Is to be filled be job._handle_params

        if self._processor.is_remote:
            job_context = None
            if converter:
                job_context = {"result_mapping": ['perceval.utils', converter.__name__]}
            payload = self._processor.prepare_job_payload(primitive)
            if self._iterator:
                payload['payload']['iterator'] = self._iterator
            payload['payload']['max_shots'] = self._max_shots
            job_name = self.default_job_name if self.default_job_name is not None else method
            return RemoteJob(payload, self._processor.get_rpc_handler(), job_name,
                             command_param_names=command_param_names,
                             delta_parameters=delta_parameters, job_context=job_context)
        else:
            func_name = f"_{primitive}_iterate_locally" if self._iterator else f"_{primitive}_wrapper"
            return LocalJob(getattr(self, func_name),
                            result_mapping_function=converter,
                            command_param_names=command_param_names,
                            delta_parameters=delta_parameters)

    @property
    def samples(self) -> Job:
        return self._create_job("samples")

    @property
    def sample_count(self) -> Job:
        return self._create_job("sample_count")

    @property
    def probs(self) -> Job:
        return self._create_job("probs")

    # Iterator construction methods
    def add_iteration(self, circuit_params: Dict = None,
                      input_state: BasicState = None,
                      min_detected_photons: int = None):
        it = {}
        if circuit_params is not None:
            it['circuit_params'] = circuit_params
        if input_state is not None:
            it['input_state'] = input_state
        if min_detected_photons is not None:
            it['min_detected_photons'] = min_detected_photons
        self._check_iteration(it)
        self._iterator.append(it)

    def _check_iteration(self, iter_params):
        assert isinstance(iter_params, dict) and iter_params, "Iteration parameters must be a valid dictionary"
        if 'circuit_params' in iter_params:
            assert isinstance(iter_params['circuit_params'], dict), \
                "Iteration: circuit_params field must be a valid dictionnary"
            for param_name, param_value in iter_params['circuit_params'].items():
                assert isinstance(param_value, Number), \
                    f"Iteration: circuit parameters have to be numerical values (got {param_value})"
                assert param_name in self._processor.get_circuit_parameters(), \
                    f"Iteration: circuit parameter {param_name} does not exist in processor"
        if 'input_state' in iter_params:
            assert isinstance(iter_params['input_state'], BasicState), \
                "Iteration: input_state field must be a basic state"
            assert iter_params['input_state'].m == self._processor.m, \
                f"Iteration: input state and processor size mismatch (processor size is {self._processor.m})"
            self._processor.check_input(iter_params['input_state'])

    def add_iteration_list(self, iterations: List[Dict]):
        for iter_params in iterations:
            self.add_iteration(**iter_params)

    def clear_iterations(self):
        # In case, the user wants to use the same sampler instance, but with a new iterator
        self._iterator = []

    @property
    def n_iterations(self):
        return len(self._iterator)

    def _probs_wrapper(self, progress_callback: Callable = None):
        # max_shots is used as the invert of the precision set in the probs computation
        # Rationale: mimic the fact that the more shots, the more accurate probability distributions are.
        precision = None if self._max_shots is None else min(1e-6, 1/self._max_shots)
        return self._processor.probs(precision, progress_callback)

    def _samples_wrapper(self, max_samples: int = None, progress_callback: Callable = None):
        if max_samples is None and self._max_shots is None:
            raise RuntimeError("Local sampling simumation requires max_samples and/or max_shots parameters")
        if max_samples is None:
            max_samples = self.SAMPLES_MAX_COUNT
        return self._processor.samples(max_samples, self._max_shots, progress_callback)


    # Local iteration methods mimic remote iterations for interchangeability purpose
    def _probs_iterate_locally(self, max_shots: int = None, progress_callback: Callable = None):
        precision = None if max_shots is None else min(1e-6, 1 / max_shots)
        results = {'results_list':[]}
        for idx, it in enumerate(self._iterator):
            self._apply_iteration(it)
            results['results_list'].append(self._processor.probs(precision))
            results['results_list'][-1]['iteration'] = it
            if progress_callback is not None:
                progress_callback((idx+1)/len(self._iterator))
        return results

    def _samples_iterate_locally(self, max_shots: int = None, max_samples: int = None, progress_callback: Callable = None):
        if max_samples is None and max_shots is None:
            raise RuntimeError("Local sampling simumation requires max_samples and/or max_shots parameters")
        if max_samples is None:
            max_samples = self.SAMPLES_MAX_COUNT
        results = {'results_list':[]}
        for idx, it in enumerate(self._iterator):
            self._apply_iteration(it)
            results['results_list'].append(self._processor.samples(max_samples, max_shots))
            results['results_list'][-1]['iteration'] = it
            if progress_callback is not None:
                progress_callback((idx+1)/len(self._iterator))
        return results

    def _apply_iteration(self, it):
        if 'circuit_params' in it:
            circuit_params = self._processor.get_circuit_parameters()
            for name, value in it['circuit_params'].items():
                circuit_params[name].set_value(value)
        if 'input_state' in it:
            self._processor.with_input(it['input_state'])
        if 'min_detected_photons' in it:
            self._processor.min_detected_photons_filter(it['min_detected_photons'])
