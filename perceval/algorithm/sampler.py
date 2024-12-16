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
from numbers import Number

from .abstract_algorithm import AAlgorithm
from perceval.utils import samples_to_sample_count, samples_to_probs, sample_count_to_samples, \
    sample_count_to_probs, probs_to_samples, probs_to_sample_count
from perceval.utils.logging import get_logger, channel
from perceval.components.abstract_processor import AProcessor
from perceval.runtime import Job, RemoteJob, LocalJob
from perceval.utils import BasicState, NoiseModel


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

    _iterator_type_check: dict[str, type] = {'circuit_params': dict,
                                             'input_state': BasicState,
                                             'min_detected_photons': int,
                                             'max_samples': int,
                                             'max_shots': int,
                                             'noise': NoiseModel}

    def __init__(self, processor: AProcessor, **kwargs):
        super().__init__(processor, **kwargs)
        self._iterator = []
        self._max_samples = None

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

    def _check_sample_shot_iterator(self) -> bool:
        return all("max_samples" in it or "max_shots" in it for it in self._iterator)

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
            delta_parameters["mapping"]['max_samples'] = None  # Is to be filled by job._handle_params
            delta_parameters["mapping"]['max_shots'] = self._max_shots
        elif method_is_probs and not primitive_is_probs:
            delta_parameters["command"]['max_samples'] = self.PROBS_SIMU_SAMPLE_COUNT
        elif not method_is_probs and not primitive_is_probs:
            delta_parameters["command"]['max_samples'] = None  # Is to be filled by job._handle_params

        if self._processor.is_remote:
            job_context = None
            if converter:
                job_context = {"result_mapping": ['perceval.utils', converter.__name__]}
            payload = self._processor.prepare_job_payload(primitive)
            if self._iterator:
                payload['payload']['iterator'] = self._iterator
            payload['payload']['max_shots'] = self._max_shots
            job_name = self.default_job_name if self.default_job_name is not None else method
            job = RemoteJob(payload, self._processor.get_rpc_handler(), job_name,
                            command_param_names=command_param_names,
                            delta_parameters=delta_parameters, job_context=job_context)
            get_logger().info(
                f"Prepare remote job (command: {primitive} on {payload['platform_name']})", channel.general)
            return job
        else:
            func_name = f"_{primitive}_iterate_locally" if self._iterator else f"_{primitive}_wrapper"
            get_logger().info(f"Prepare local job (command: Sampler.{func_name})", channel.general)
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
    def _add_iteration(self, iter_params):
        self._check_iteration(iter_params)
        self._iterator.append(iter_params)

    def _check_iteration(self, iter_params):
        assert isinstance(iter_params, dict), "Iteration parameters must be a valid dictionary"
        for key, val in iter_params.items():
            if key in self._iterator_type_check:
                correct_type = self._iterator_type_check[key]
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
                    assert param_name in self._processor.get_circuit_parameters(), \
                        f"Iteration: circuit parameter {param_name} does not exist in processor"
            elif key == 'input_state':
                assert val.m == self._processor.m, \
                    f"Iteration: input state and processor size mismatch (processor size is {self._processor.m})"
                self._processor.check_input(iter_params['input_state'])

    def add_iteration(self, **kwargs):
        """
        Currently accepted keywords:

        - circuit_params: dict containing pairs (parameter_name: str - value : number)
        - input_state: BasicState
        - min_detected_photons: int
        - max_samples: int
        - max_shots: int
        - noise: NoiseModel
        """
        get_logger().info("Add 1 iteration to Sampler", channel.general)
        self._add_iteration(kwargs)

    def add_iteration_list(self, iterations: list[dict]):
        get_logger().info(f"Add {len(iterations)} iterations to Sampler", channel.general)
        for iter_params in iterations:
            self._add_iteration(iter_params)

    def clear_iterations(self):
        # In case, the user wants to use the same sampler instance, but with a new iterator
        get_logger().info(f"Clear all iterations in Sampler", channel.general)
        self._iterator = []

    @property
    def n_iterations(self):
        return len(self._iterator)

    def _probs_wrapper(self, progress_callback: callable = None):
        # max_shots is used as the invert of the precision set in the probs computation
        # Rationale: mimic the fact that the more shots, the more accurate probability distributions are.
        precision = None if self._max_shots is None else min(1e-6, 1 / self._max_shots)
        return self._processor.probs(precision, progress_callback)

    def _samples_wrapper(self, max_samples: int = None, progress_callback: callable = None):
        if max_samples is None and self._max_shots is None:
            raise RuntimeError("Local sampling simulation requires max_samples and/or max_shots parameters")
        if max_samples is None:
            max_samples = self.SAMPLES_MAX_COUNT
        return self._processor.samples(max_samples, self._max_shots, progress_callback)

    # Local iteration methods mimic remote iterations for interchangeability purpose
    def _probs_iterate_locally(self, max_shots: int = None, progress_callback: callable = None):
        self._max_shots = max_shots
        default_it = self._it_default_parameters()
        results = {'results_list': []}
        for idx, it in enumerate(self._iterator):
            self._processor._simulator = None  # Reset any possible cached parameter
            self._apply_iteration(default_it | it)
            precision = None if self._max_shots is None else min(1e-6, 1 / self._max_shots)
            results['results_list'].append(self._processor.probs(precision))
            results['results_list'][-1]['iteration'] = it
            if progress_callback is not None:
                progress_callback((idx + 1) / len(self._iterator))
        self._apply_iteration(default_it)
        return results

    def _samples_iterate_locally(self, max_shots: int = None, max_samples: int = None,
                                 progress_callback: callable = None):
        if max_samples is None and max_shots is None:
            if not self._check_sample_shot_iterator():
                raise RuntimeError("Local sampling simulation requires max_samples and/or max_shots parameters")

        if max_samples is None:
            max_samples = self.SAMPLES_MAX_COUNT
        self._max_samples = max_samples
        self._max_shots = max_shots
        default_it = self._it_default_parameters()
        results = {'results_list': []}
        for idx, it in enumerate(self._iterator):
            self._processor._simulator = None  # Reset any possible cached parameter
            self._apply_iteration(default_it | it)
            results['results_list'].append(self._processor.samples(self._max_samples, self._max_shots))
            results['results_list'][-1]['iteration'] = it
            if progress_callback is not None:
                progress_callback((idx + 1) / len(self._iterator))
        self._apply_iteration(default_it)  # restore default parameters
        return results

    def _apply_iteration(self, it):
        for key, val in it.items():
            try:
                self.__getattribute__(f"_set_{key}")(val)
            except AttributeError:
                pass

    def _set_circuit_params(self, params: dict):
        if params:
            circuit_params = self._processor.get_circuit_parameters()
            for name, value in params.items():
                if value is not None:
                    circuit_params[name].set_value(value)

    def _set_input_state(self, input_state: BasicState):
        self._processor.with_input(input_state)

    def _set_min_detected_photons(self, count: int):
        self._processor.min_detected_photons_filter(count)
        if count is None:
            self._processor.parameters.pop("min_detected_photons")

    def _set_max_samples(self, val: int):
        self._max_samples = val

    def _set_max_shots(self, val: int):
        self._max_shots = val

    def _set_noise(self, noise: NoiseModel):
        self._processor.noise = noise

    def _it_default_parameters(self) -> dict:
        """Creates an iteration with default parameters"""
        input_state = self._processor.input_state
        if isinstance(input_state, BasicState) and self._processor.heralds:
            # If it's not a BasicState, the user needs to provide all the modes
            input_state = BasicState([v for m, v in enumerate(input_state) if m not in self._processor.heralds])

        return {"circuit_params": {k: v._value for k, v in self._processor.get_circuit_parameters().items()},
                "input_state": input_state,
                "min_detected_photons": self._processor.parameters.get("min_detected_photons", None),
                "max_samples": self._max_samples,
                "max_shots": self._max_shots,
                "noise": self._processor.noise
                }
