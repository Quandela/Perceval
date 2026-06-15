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

from perceval.backends import ABackend, AStrongSimulationBackend, ExqaliburBackendWrapper, BACKEND_LIST
from perceval.components import Experiment, Source
from perceval.simulators import SimulatorFactory, ExqaliburNoisySamplingSimulator, NoisySamplingSimulator
from perceval.utils import NoiseModel, BasicState, StateVector, SVDistribution, AnnotatedFockState, ProcessorType, \
    ConversionHelper, ProgressCallback, noise_to_perf_dict
from perceval.utils.logging import get_logger, channel

from .local_computer import LocalComputer
from .computation import Computation
from .platform_specs import PlatformSpecs


class SimulatedComputer(LocalComputer):
    """
    A computer able to perform local simulations
    :param backend: The backend to use to perform the simulations. Can be a backend name or a backend instance
    """

    PROBS_DEFAULT_SAMPLES = 10000

    def __init__(self, backend):
        super().__init__()
        self._init_backend(backend)
        self._noise: NoiseModel = NoiseModel()

        cls = type(self)
        self._register_method(cls.probs)
        self._register_method(cls.samples)
        self._register_method(cls.sample_count)

    def _init_backend(self, backend):
        if isinstance(backend, str):
            assert backend in BACKEND_LIST, f"Unknown simulation backend '{backend}'. Possible backends: {BACKEND_LIST}"
            self._backend = BACKEND_LIST[backend]()
        else:
            assert isinstance(backend, ABackend), f"'backend' must be an ABackend (got {type(backend).__name__})"
            self._backend = backend

    @property
    def noise(self) -> NoiseModel:
        return self._noise

    @noise.setter
    def noise(self, noise: NoiseModel):
        self._noise = noise

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
        if experiment.noise is None:
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
    def _parse_precision(precision: float = None, max_shots: int = None, max_samples: int = None) -> float | None:
        if precision is not None:
            return precision
        nb_shots = max_shots or max_samples
        return None if nb_shots is None else min(1e-6, 1 / nb_shots)

    def probs(self, experiment: Experiment,
              progress_callback: ProgressCallback = None,
              precision: float = None,
              max_samples: int = None,
              max_shots: int = None,
              compilation_seed: int = None,
              **kwargs) -> dict:
        """
        Computes the probabilities for a given experiment. Does not apply error mitigations
        :param experiment: The Experiment to simulate.
        :param precision: The precision of the computation.
         Probabilities lower than the biggest input probability times this are ignored. Used only with Probability backends
        :param max_shots: The maximum number of shots to consider. A shot is any event with at least 1 photon
         Used only is the computer has a Sampling backend or if the precision is not given
        :param max_samples. The maximum number of samples to consider.
         A sample is any event with at least min_photons photon (defined in the Experiment).
         Used only is the computer has a Sampling backend or if the precision and the max_shots are not given
        :param compilation_seed: A seed to use for the compilation starting point or the random phases
        :return:
        """
        if isinstance(self._backend, AStrongSimulationBackend):
            experiment = experiment.use_phase_noise(self.noise, compilation_seed)
            simulator = SimulatorFactory.build(experiment, self._backend)

            precision = self._parse_precision(precision, max_shots, max_samples)
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

        if max_samples is None:
            max_samples = self.PROBS_DEFAULT_SAMPLES

        res = self.sample_count(experiment, max_samples, max_shots, progress_callback, compilation_seed, **kwargs)
        res["results"] = ConversionHelper.convert_to("probs", res["results"])
        return res

    def _setup_sampling_simulator(self, experiment: Experiment, compilation_seed: int | None):
        if isinstance(self._backend, ExqaliburBackendWrapper):
            simulator = ExqaliburNoisySamplingSimulator(self._backend)
        else:
            simulator = NoisySamplingSimulator(self._backend)
        simulator.sleep_between_batches = 0  # Remove sleep time between batches of samples in local simulation

        experiment = experiment.use_phase_noise(self.noise, compilation_seed)
        simulator.set_circuit(experiment.unitary_circuit())
        simulator.set_selection(
            min_detected_photons_filter=experiment.min_photons_filter,
            postselect=experiment.post_select_fn,
            heralds=experiment.heralds)
        simulator.keep_heralds(False)
        simulator.compute_physical_logical_perf(self._parameters.get("compute_physical_logical_perf", False))
        simulator.set_detectors(experiment.detectors)

        source = self._create_source(experiment)
        get_logger().info(f"Start a local {'perfect' if source.is_perfect() else 'noisy'} sampling", channel.general)
        return simulator, self._make_input(experiment, source)

    def samples(self, experiment: Experiment,
                max_samples: int,
                max_shots: int = None,
                progress_callback: ProgressCallback = None,
                compilation_seed: int = None,
                **kwargs) -> dict:
        if isinstance(self._backend, AStrongSimulationBackend):
            res = self.probs(experiment,
                             progress_callback,
                             max_samples=max_samples,
                             max_shots=max_shots,
                             compilation_seed=compilation_seed,
                             **kwargs)
            res["results"] = ConversionHelper.convert_to("samples", res["results"], max_samples=max_samples, max_shots=max_shots, **kwargs)
            return res

        self.log_resources(sys._getframe().f_code.co_name, experiment, {'max_samples': max_samples, 'max_shots': max_shots})

        simulator, sample_provider = self._setup_sampling_simulator(experiment, compilation_seed)
        res = simulator.samples(sample_provider, max_samples, max_shots, progress_callback)
        get_logger().info("Local sampling complete!", channel.general)
        return res

    def sample_count(self, experiment: Experiment,
                     max_samples: int,
                     max_shots: int = None,
                     progress_callback: ProgressCallback = None,
                     compilation_seed: int = None,
                     **kwargs) -> dict:
        if isinstance(self._backend, AStrongSimulationBackend):
            res = self.probs(experiment,
                             progress_callback,
                             max_samples=max_samples,
                             max_shots=max_shots,
                             compilation_seed=compilation_seed,
                             **kwargs)
            res["results"] = ConversionHelper.convert_to("sample_count", res["results"], max_samples=max_samples, max_shots=max_shots, **kwargs)
            return res

        self.log_resources(sys._getframe().f_code.co_name, experiment, {'max_samples': max_samples, 'max_shots': max_shots})

        simulator, sample_provider = self._setup_sampling_simulator(experiment, compilation_seed)
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
