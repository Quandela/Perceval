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

from .abstract_algorithm import AAlgorithm
from .parameter_iterator import ParameterIterator

from perceval.utils import samples_to_sample_count, samples_to_probs, sample_count_to_samples, \
    sample_count_to_probs, probs_to_samples, probs_to_sample_count
from perceval.utils.logging import get_logger, channel
from perceval.runtime.abstract_processor import AProcessor
from perceval.runtime import Job, RemoteJob, LocalJob


class Sampler(AAlgorithm):
    """
    Basic algorithm able to retrieve some sampling data via 3 main methods.

    * samples(max_samples): returns a list of sampled states
    * sample_count(max_samples): return a mapping {(output state) : (sampled count)}
    * probs(): returns a probability distribution of output states

    The form of the output for all 3 sampling methods is a dictionary containing a 'results' key and several performance
    values.

    The Sampler also supports batch jobs, where several computations can be stored in a single job.

    :param processor: the processor to sample on
    :param kwargs: when the processor is remote, a "max_shots_per_call" should be passed in order to
                   limit the shots usage.
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
        self._max_samples = None
        self._iterator = ParameterIterator(processor.experiment, self._max_shots, self._max_samples)

    def _input_available(self) -> bool:
        if not self._iterator.input_available():
            return False

        # Further checks using the Processor
        try:
            for iter_params in self._iterator.iterations:
                if 'input_state' in iter_params:
                    self._processor.check_input(iter_params['input_state'])
            return True
        except Exception as e:
            get_logger().error(e, channel.user)
            return False

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
        """Create a sample stream job

        :return: A job where chronologically ordered samples are expected as result
        """
        return self._create_job("samples")

    @property
    def sample_count(self) -> Job:
        """Create a sample count job

        :return: A job where sample counts are expected as result
        """
        return self._create_job("sample_count")

    @property
    def probs(self) -> Job:
        """Create a sample count job

        :return: A job where sample counts are expected as result
        """
        return self._create_job("probs")

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
        get_logger().info("Add 1 iteration to Sampler", channel.general)
        self._iterator.add_iteration(**kwargs)

    def add_iteration_list(self, iterations: list[dict]):
        """
        Add multiple iterations to future jobs.
        """
        get_logger().info(f"Add {len(iterations)} iterations to Sampler", channel.general)
        for iter_params in iterations:
            self._iterator.add_iteration(**iter_params)

    def clear_iterations(self):
        """
        Clear all prepared iterations.
        """
        get_logger().info(f"Clear all iterations in Sampler", channel.general)
        self._iterator.clear_iterations()

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
        default_experiment = self._processor.experiment

        results = {'results_list': []}
        for idx, it in enumerate(self._iterator):
            max_shots_local = max_shots or it.max_shots
            precision = None if max_shots_local is None else min(1e-6, 1 / max_shots_local)
            self._processor.experiment = it.experiment
            results['results_list'].append(self._processor.probs(precision))
            results['results_list'][-1]['iteration'] = it.parameters
            if progress_callback is not None:
                progress_callback((idx + 1) / len(self._iterator))

        self._processor.experiment = default_experiment

        return results

    def _samples_iterate_locally(self, max_shots: int = None, max_samples: int = None,
                                 progress_callback: callable = None):
        if max_samples is None and max_shots is None and not self._iterator.check_sample_shot_iterator():
            raise RuntimeError("Local sampling simulation requires max_samples and/or max_shots parameters")

        if max_samples is None:
            max_samples = self.SAMPLES_MAX_COUNT

        default_experiment = self._processor.experiment

        results = {'results_list': []}
        for idx, it in enumerate(self._iterator):
            max_samples_local = it.max_samples or max_samples
            self._processor.experiment = it.experiment
            results['results_list'].append(self._processor.samples(max_samples_local, it.max_shots))
            results['results_list'][-1]['iteration'] = it.parameters
            if progress_callback is not None:
                progress_callback((idx + 1) / len(self._iterator))

        self._processor.experiment = default_experiment

        return results
