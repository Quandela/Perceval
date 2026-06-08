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

import sys

from perceval.backends import AStrongSimulationBackend, ExqaliburBackendWrapper
from perceval.components import Experiment, Source
from perceval.simulators import SimulatorFactory, ExqaliburNoisySamplingSimulator, NoisySamplingSimulator
from perceval.simulators.noisy_sampling_simulator import ASamplingSimulator
from perceval.utils import NoiseModel, BasicState, StateVector, SVDistribution, AnnotatedFockState, ProcessorType, \
    ConversionHelper, ProgressCallback, noise_to_perf_dict
from perceval.utils.logging import get_logger, channel

from .local_computer import LocalComputer
from .computation import Computation
from .platform_specs import PlatformSpecs


class SimulatedComputer(LocalComputer):

    PROBS_DEFAULT_SAMPLES = 10000

    def __init__(self, backend):
        super().__init__()
        self._init_backend(backend)
        self._has_custom_noise = False  # Legacy; allows noise to be defined in Experiment if False
        self._noise: NoiseModel = NoiseModel()

    def _init_backend(self, backend):
        if isinstance(backend, str):
            from perceval import BACKEND_LIST
            assert backend in BACKEND_LIST, f"Unknown simulation backend '{backend}'. Possible backends: {BACKEND_LIST}"
            self._backend = BACKEND_LIST[backend]()
        else:
            from perceval import ABackend
            assert isinstance(backend, ABackend), f"'backend' must be an ABackend (got {type(backend).__name__})"
            self._backend = backend

    @property
    def noise(self) -> NoiseModel:
        return self._noise

    @noise.setter
    def noise(self, noise: NoiseModel | None):
        if noise is not None:
            self._has_custom_noise = True
            self._noise = noise
        else:
            self._has_custom_noise = False
            self._noise = NoiseModel()

    def validate_single(self, computation: Computation) -> None:
        super().validate_single(computation)
        self.check_min_detected_photons_filter(computation)

    @property
    def specs(self) -> PlatformSpecs:
        res = PlatformSpecs()
        res.parameters = self.available_parameters
        return res

    @property
    def available_parameters(self) -> dict[str, str]:
        return {"compute_physical_logical_perf": "bool. If True, physical and logical performances will be returned."
                                                 "Else, only a global performance will be returned."}

    def _create_source(self, experiment: Experiment) -> Source:
        if self._has_custom_noise or experiment.noise is None:
            return Source.from_noise_model(self.noise)
        return Source.from_noise_model(experiment.noise)

    def check_min_detected_photons_filter(self, computation: Computation) -> None:
        experiment = computation.experiment
        if experiment.min_photons_filter is None:
            source = self._create_source(experiment)
            # Automatically set the min_photons_filter for perfect sources if not set
            if source.is_perfect() and isinstance(experiment.input_state, BasicState):
                experiment.min_detected_photons_filter(experiment.input_state.n - sum(experiment.heralds.values()))
            else:
                raise ValueError("The value of min_detected_photons is not set."
                                 " Use the method experiment.min_detected_photons_filter(value).")

    @staticmethod
    def _make_input(experiment: Experiment, source: Source):
        if isinstance(experiment.input_state, SVDistribution) \
                or (isinstance(experiment.input_state, AnnotatedFockState) and experiment.input_state.has_polarization):
            # Custom input
            return experiment.input_state

        return source, experiment.input_state

    @staticmethod
    def _parse_precision(kwargs) -> float | None:
        if "precision" in kwargs:
            return kwargs["precision"]
        nb_shots = kwargs.get("max_shots", kwargs.get("max_samples", None))
        return None if nb_shots is None else min(1e-6, 1 / nb_shots)

    def probs(self, experiment: Experiment, progress_callback: ProgressCallback = None, **kwargs) -> dict:
        """
        Computes the probabilities for a given experiment. Does not apply error mitigations
        :param experiment:
        :param kwargs:
        :return:
        """
        if isinstance(self._backend, AStrongSimulationBackend):
            simulator = SimulatorFactory.build(experiment, self._backend)

            precision = self._parse_precision(kwargs)
            if precision is not None:
                simulator.set_precision(precision)
            source = self._create_source(experiment)
            get_logger().info(f"Start a local {'perfect' if source.is_perfect() else 'noisy'} strong simulation",
                              channel.general)
            simulator.keep_heralds(False)
            simulator.compute_physical_logical_perf(self._parameters.get("compute_physical_logical_perf", False))
            svd = self._make_input(experiment, source)
            res = simulator.probs_svd(svd, experiment.detectors, progress_callback)
            get_logger().info("Local strong simulation complete!", channel.general)

            self.log_resources(sys._getframe().f_code.co_name, experiment, {'precision': precision})
            return res

        if "max_samples" not in kwargs:
            kwargs["max_samples"] = self.PROBS_DEFAULT_SAMPLES

        res = self.sample_count(experiment, progress_callback, **kwargs)
        res["results"] = ConversionHelper.convert_to("probs", res["results"])
        return res

    def _setup_sampling_simulator(self, experiment: Experiment) -> ASamplingSimulator:
        if isinstance(self._backend, ExqaliburBackendWrapper):
            simulator = ExqaliburNoisySamplingSimulator(self._backend)
        else:
            simulator = NoisySamplingSimulator(self._backend)
        simulator.sleep_between_batches = 0  # Remove sleep time between batches of samples in local simulation
        # TODO: solve discrepancy for phase noise (SimulatorFactory.build)
        simulator.set_circuit(experiment.unitary_circuit())
        simulator.set_selection(
            min_detected_photons_filter=experiment.min_photons_filter,
            postselect=experiment.post_select_fn,
            heralds=experiment.heralds)
        simulator.keep_heralds(False)
        simulator.compute_physical_logical_perf(self._parameters.get("compute_physical_logical_perf", False))
        simulator.set_detectors(experiment.detectors)
        return simulator

    def samples(self, experiment: Experiment, progress_callback: ProgressCallback = None, **kwargs) -> dict:
        if isinstance(self._backend, AStrongSimulationBackend):
            res = self.probs(experiment, progress_callback, **kwargs)
            res["results"] = ConversionHelper.convert_to("samples", res["results"], **kwargs)
            return res

        max_samples = kwargs["max_samples"]
        max_shots = kwargs.get("max_shots", None)
        simulator = self._setup_sampling_simulator(experiment)
        self.log_resources(sys._getframe().f_code.co_name, experiment, {'max_samples': max_samples, 'max_shots': max_shots})
        source = self._create_source(experiment)
        get_logger().info(f"Start a local {'perfect' if source.is_perfect() else 'noisy'} sampling", channel.general)
        sample_provider = self._make_input(experiment, source)
        res = simulator.samples(sample_provider, max_samples, max_shots, progress_callback)
        get_logger().info("Local sampling complete!", channel.general)
        return res

    def sample_count(self, experiment: Experiment, progress_callback: ProgressCallback = None, **kwargs) -> dict:
        if isinstance(self._backend, AStrongSimulationBackend):
            res = self.probs(experiment, progress_callback, **kwargs)
            res["results"] = ConversionHelper.convert_to("sample_count", res["results"], **kwargs)
            return res

        max_samples = kwargs["max_samples"]
        max_shots = kwargs.get("max_shots", None)
        simulator = self._setup_sampling_simulator(experiment)
        self.log_resources(sys._getframe().f_code.co_name, experiment,
                           {'max_samples': max_samples, 'max_shots': max_shots})
        source = self._create_source(experiment)
        get_logger().info(f"Start a local {'perfect' if source.is_perfect() else 'noisy'} sampling", channel.general)
        sample_provider = self._make_input(experiment, source)
        res = simulator.sample_count(sample_provider, max_samples, max_shots, progress_callback)
        get_logger().info("Local sampling complete!", channel.general)
        return res

    def log_resources(self, method: str, experiment: Experiment, extra_parameters: dict):
        """Log resources of the AbstractComputer

        :param method: name of the method used
        :param extra_parameters: extra parameters to log.

            Extra parameter can be:

                - max_samples
                - max_shots
                - precision
        """
        extra_parameters = {key: value for key, value in extra_parameters.items() if value is not None}
        my_dict = {
            'layer': type(self).__name__,
            'backend': self._backend.name,
            'm': experiment.circuit_size,
            'method': method
        }
        if isinstance(experiment.input_state, BasicState):
            my_dict['n'] = experiment.input_state.n
        elif isinstance(experiment.input_state, StateVector):
            my_dict['n'] = max(experiment.input_state.n)
        elif isinstance(experiment.input_state, SVDistribution):
            my_dict['n'] = experiment.input_state.n_max
        else:
            get_logger().error(f"Cannot get n for type {type(experiment.input_state).__name__}", channel.general)
        if extra_parameters:
            my_dict.update(extra_parameters)
        if self.noise != NoiseModel():
            my_dict['noise'] = self.noise.__dict__()
        get_logger().log_resources(my_dict)

    def compute_physical_logical_perf(self, value: bool):
        """
        Tells the simulator to compute or not the physical and logical performances when possible

        :param value: True to compute the physical and logical performances, False otherwise.
        """
        self._parameters["compute_physical_logical_perf"] = value

    @property
    def type(self):
        return ProcessorType.SIMULATOR

    @property
    def performance(self):
        return noise_to_perf_dict(self.noise)
