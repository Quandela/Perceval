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
from __future__ import annotations

import sys

from perceval.backends import ABackend, ASamplingBackend, BACKEND_LIST
from perceval.utils import SVDistribution, BasicState, StateVector, NoiseModel
from perceval.utils.logging import get_logger, channel

from .abstract_processor import AProcessor, ProcessorType
from .experiment import Experiment
from .linear_circuit import ACircuit, Circuit
from .source import Source
from .unitary_components import PS

class Processor(AProcessor):
    """
    Generic definition of processor as an experiment + simulation backend

    :param backend: Name or instance of a simulation backend
    :param m_circuit: can either be:

        * an int: number of modes of interest (MOI). A mode of interest is any non-heralded mode.
            >>> p = Processor("SLOS", 5)

        * a circuit: the input circuit to start with. Other components can still be added afterwards with `add()`
            >>> p = Processor("SLOS", BS() // PS() // BS())

        * an experiment:
            >>> p = Processor("SLOS", Experiment(BS(), NoiseModel(0.8)))

    :param noise: a NoiseModel containing noise parameters (defaults to no noise)
    :param name: a textual name for the processor (defaults to "Local processor")
    """
    def __init__(self, backend: ABackend | str, m_circuit: int | ACircuit | Experiment = None,
                 noise: NoiseModel = None, name: str = "Local processor"):
        if not isinstance(m_circuit, Experiment):
            m_circuit = Experiment(m_circuit, noise=noise, name=name)
        super().__init__(m_circuit)

        self._init_backend(backend)
        self._previous_noise = None
        self._inputs_map = None
        self._noise_changed_observer()
        self._input_changed_observer()
        self._simulator = None

    @property
    def _has_custom_input(self):
        return (isinstance(self.input_state, SVDistribution)
                or (isinstance(self.input_state, BasicState) and self.input_state.has_polarization))

    def _noise_changed_observer(self):
        self._source = Source.from_noise_model(self.noise)
        if not self._has_custom_input:
            self._inputs_map = None
        self._previous_noise = self.noise

    @AProcessor.noise.getter
    def noise(self):
        noise = super(Processor, type(self)).noise.fget(self)
        if noise is None:
            return NoiseModel()
        return noise

    @property
    def source_distribution(self) -> SVDistribution | None:
        r"""
        Retrieve the computed input distribution. Compute it if it is not cached and an input state has been provided.
        :return: the input SVDistribution if `with_input` was called previously, otherwise None.
        """
        if self._inputs_map is None and self.input_state is not None:
            self._generate_noisy_input()
        return self._inputs_map

    def _circuit_change_observer(self, new_component = None):
        self._simulator = None

    @property
    def source(self):
        """
        :return: The photonic source
        """
        return self._source

    def _init_backend(self, backend):
        if isinstance(backend, str):
            assert backend in BACKEND_LIST, f"Simulation backend '{backend}' does not exist"
            self.backend = BACKEND_LIST[backend]()
        else:
            assert isinstance(backend, ABackend), f"'backend' must be an ABackend (got {type(backend)})"
            self.backend = backend

    def type(self) -> ProcessorType:
        return ProcessorType.SIMULATOR

    @property
    def is_remote(self) -> bool:
        return False

    def _generate_noisy_input(self):
        self._inputs_map = self._source.generate_distribution(self.input_state)

    def generate_noisy_heralds(self) -> SVDistribution:
        if self.heralds:
            heralds_perfect_state = BasicState([v for k, v in sorted(self.heralds.items())])
            return self._source.generate_distribution(heralds_perfect_state)
        return SVDistribution()

    def _input_changed_observer(self):
        if isinstance(self.input_state, BasicState):
            if self.input_state.has_polarization:
                self._inputs_map = SVDistribution(self.input_state)
            else:
                self._generate_noisy_input()
        elif isinstance(self.input_state, SVDistribution):
            self._inputs_map = self.input_state

    def with_polarized_input(self, bs: BasicState):
        self.experiment.with_polarized_input(bs)

    def clear_input_and_circuit(self, new_m=None):
        super().clear_input_and_circuit(new_m)
        self._inputs_map = None

    def linear_circuit(self, flatten: bool = False) -> Circuit:
        """
        Creates a linear circuit from internal components, if all internal components are unitary. Takes phase
        imprecision noise into account.

        :param flatten: if True, the component recursive hierarchy is discarded, making the output circuit "flat".
        :raises RuntimeError: If any component is non-unitary
        :return: The resulting Circuit object
        """
        return self.experiment.unitary_circuit(flatten=flatten, use_phase_noise=True)

    def samples(self, max_samples: int, max_shots: int = None, progress_callback=None) -> dict:
        self.check_min_detected_photons_filter()
        from perceval.simulators import NoisySamplingSimulator
        assert isinstance(self.backend, ASamplingBackend), "A sampling backend is required to call samples method"
        sampling_simulator = NoisySamplingSimulator(self.backend)
        sampling_simulator.sleep_between_batches = 0  # Remove sleep time between batches of samples in local simulation
        sampling_simulator.set_circuit(self.linear_circuit())
        sampling_simulator.set_selection(
            min_detected_photons_filter=self._min_detected_photons_filter,
            postselect=self.post_select_fn,
            heralds=self.heralds)
        sampling_simulator.keep_heralds(False)
        sampling_simulator.set_detectors(self.detectors)
        self.log_resources(sys._getframe().f_code.co_name, {'max_samples': max_samples, 'max_shots': max_shots})
        get_logger().info(
            f"Start a local {'perfect' if self._source.is_perfect() else 'noisy'} sampling", channel.general)
        sample_provider = self.source_distribution if self._has_custom_input else (self._source, self.input_state)
        res = sampling_simulator.samples(sample_provider, max_samples, max_shots, progress_callback)
        get_logger().info("Local sampling complete!", channel.general)
        return res

    def probs(self, precision: float = None, progress_callback: callable = None) -> dict:
        self.check_min_detected_photons_filter()

        # assert self._inputs_map is not None, "Input is missing, please call with_inputs()"
        if self._simulator is None:
            from perceval.simulators import SimulatorFactory  # Avoids a circular import
            self._simulator = SimulatorFactory.build(self)
        else:
            self._simulator.set_circuit(self.linear_circuit() if self.experiment.is_unitary else self.components, self.circuit_size)
            self._simulator.set_min_detected_photons_filter(self._min_detected_photons_filter)

        if precision is not None:
            self._simulator.set_precision(precision)
        get_logger().info(f"Start a local {'perfect' if self._source.is_perfect() else 'noisy'} strong simulation",
                          channel.general)
        self._simulator.keep_heralds(False)
        res = self._simulator.probs_svd(self.source_distribution, self.detectors, progress_callback)
        get_logger().info("Local strong simulation complete!", channel.general)

        self.log_resources(sys._getframe().f_code.co_name, {'precision': precision})
        return res

    @property
    def available_commands(self) -> list[str]:
        return ["samples" if isinstance(self.backend, ASamplingBackend) else "probs"]

    def log_resources(self, method: str, extra_parameters: dict):
        """Log resources of the processor

        :param method: name of the method used
        :param extra_parameters: extra parameters to log.

            Extra parameter can be:

                - max_samples
                - max_shots
                - precision
        """
        extra_parameters = {key: value for key, value in extra_parameters.items() if value is not None}
        my_dict = {
            'layer': 'Processor',
            'backend': self.backend.name,
            'm': self.circuit_size,
            'method': method
        }
        if isinstance(self.input_state, BasicState):
            my_dict['n'] = self.input_state.n
        elif isinstance(self.input_state, StateVector):
            my_dict['n'] = max(self.input_state.n)
        elif isinstance(self.input_state, SVDistribution):
            my_dict['n'] = self.input_state.n_max
        else:
            get_logger().error(f"Cannot get n for type {type(self.input_state).__name__}", channel.general)
        if extra_parameters:
            my_dict.update(extra_parameters)
        if self.noise != NoiseModel():
            my_dict['noise'] = self.noise.__dict__()
        get_logger().log_resources(my_dict)
