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

from perceval.utils import SVDistribution, BasicState, FockState, AnnotatedFockState, StateVector, NoiseModel, \
                           ProcessorType, ProgressCallback
from perceval.utils.logging import get_logger, channel

from perceval.components.experiment import Experiment
from perceval.components.linear_circuit import ACircuit, Circuit
from perceval.components.source import Source

from .abstract_processor import AProcessor
from ..simulated_computer import SimulatedComputer


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
    def __init__(self, backend, m_circuit: int | ACircuit | Experiment = None,
                 noise: NoiseModel = None, name: str = "Local processor"):
        if not isinstance(m_circuit, Experiment):
            m_circuit = Experiment(m_circuit, noise=noise, name=name)
        elif noise:
            m_circuit = m_circuit.copy()  # Create a copy so that we don't change the input experiment
        super().__init__(m_circuit)

        self._init_backend(backend)  # the only reason to keep this is that backend is public
        self._computer = SimulatedComputer(self.backend)

        if noise is not None:
            self._computer.noise = noise
        self._noise_changed_observer()

    def _noise_changed_observer(self):
        self._source = None

    @property
    def noise(self):
        noise = super(Processor, type(self)).noise.fget(self)
        if noise is None:
            return self._computer.noise
        return noise

    @noise.setter
    def noise(self, noise: NoiseModel):
        if self._experiment.noise is not None:
            self._experiment.noise = noise
        self._computer.noise = noise

    @property
    def source_distribution(self) -> SVDistribution | None:
        r"""
        Retrieve the computed input distribution. Compute it if it is not cached and an input state has been provided.
        :return: the input SVDistribution if `with_input` was called previously, otherwise None.
        """
        if isinstance(self.input_state, FockState):
            return self.source.generate_distribution(self.input_state)
        return self._experiment.input_state

    @property
    def source(self):
        """
        :return: The photonic source
        """
        if self._source is None:
            self._source = Source.from_noise_model(self.noise)
        return self._source

    def _init_backend(self, backend):
        if isinstance(backend, str):
            from perceval import BACKEND_LIST
            assert backend in BACKEND_LIST, f"Simulation backend '{backend}' does not exist"
            self.backend = BACKEND_LIST[backend]()
        else:
            from perceval import ABackend
            assert isinstance(backend, ABackend), f"'backend' must be an ABackend (got {type(backend)})"
            self.backend = backend

    def type(self) -> ProcessorType:
        return ProcessorType.SIMULATOR

    @property
    def is_remote(self) -> bool:
        return False

    def generate_noisy_heralds(self) -> SVDistribution:
        if self.in_heralds:
            heralds_perfect_state = FockState([v for k, v in sorted(self.experiment.in_heralds.items())])
            return self.source.generate_distribution(heralds_perfect_state)
        return SVDistribution()

    def linear_circuit(self, flatten: bool = False) -> Circuit:
        """
        Creates a linear circuit from internal components, if all internal components are unitary. Takes phase
        imprecision noise into account.

        :param flatten: if True, the component recursive hierarchy is discarded, making the output circuit "flat".
        :raises RuntimeError: If any component is non-unitary
        :return: The resulting Circuit object
        """
        experiment = self.experiment.use_phase_noise(self.noise)
        return experiment.unitary_circuit(flatten=flatten)

    def samples(self, max_samples: int, max_shots: int = None, progress_callback=None) -> dict:
        # Experiment's noise takes precedence
        with self._computer.apply_configuration(noise = self.experiment.noise):
            return self._computer.samples(self.experiment, max_samples=max_samples, max_shots=max_shots, progress_callback=progress_callback)

    def probs(self, precision: float = None, progress_callback: ProgressCallback = None) -> dict:
        # Experiment's noise takes precedence
        with self._computer.apply_configuration(noise=self.experiment.noise):
            return self._computer.probs(self.experiment, precision=precision, progress_callback=progress_callback)

    @property
    def available_commands(self) -> list[str]:
        from perceval.backends import ASamplingBackend
        return ["samples" if isinstance(self.backend, ASamplingBackend) else "probs"]

    def compute_physical_logical_perf(self, value: bool):
        """
        Tells the simulator to compute or not the physical and logical performances when possible

        :param value: True to compute the physical and logical performances, False otherwise.
        """
        self._computer.compute_physical_logical_perf(value)
