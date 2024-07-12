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

from deprecated import deprecated
from multipledispatch import dispatch
from numpy import inf
from typing import Dict, Callable, Union, List

from .abstract_processor import AProcessor, ProcessorType
from .source import Source
from .linear_circuit import ACircuit, Circuit
from perceval.utils import SVDistribution, BSDistribution, BasicState, StateVector, LogicalState, NoiseModel
from perceval.backends import ABackend, ASamplingBackend, BACKEND_LIST



class Processor(AProcessor):
    """
    Generic definition of processor as a source + components (both unitary and non-unitary) + ports
    + optional post-processing logic

    :param backend: Name or instance of a simulation backend
    :param m_circuit: can either be:

        * an int: number of modes of interest (MOI). A mode of interest is any non-heralded mode.
            >>> p = Processor("SLOS", 5)

        * a circuit: the input circuit to start with. Other components can still be added afterwards with `add()`
            >>> p = Processor("SLOS", BS() // PS() // BS())

    :param source: the Source used by the processor (defaults to perfect source)
    :param noise: a NoiseModel containing noise parameters (defaults to no noise)
                  Note: source and noise are mutually exclusive
    :param name: a textual name for the processor (defaults to "Local processor")
    """
    def __init__(self, backend: Union[ABackend, str], m_circuit: Union[int, ACircuit] = None, source: Source = None,
                 noise: NoiseModel = None, name: str = "Local processor"):
        super().__init__()
        self._init_backend(backend)
        self._init_circuit(m_circuit)
        self._init_noise(noise, source)
        self.name = name
        self._inputs_map: Union[SVDistribution, None] = None
        self._simulator = None

    def _init_noise(self, noise: NoiseModel, source: Source):
        self._phase_quantization = 0  # Default = infinite precision

        # Backward compatibility case: the user passes a Source
        if source is not None:
            # If he also passed noise parameters: conflict between noise parameters => raise an exception
            if noise is not None:
                raise ValueError("Both 'source' and 'noise' parameters were set. You should only input a NoiseModel")
            self.source = source

        # The user passes a NoiseModel
        elif noise is not None:
            self.noise = noise

        # Default = perfect simulation
        else:
            self._source = Source()

    @AProcessor.noise.setter
    def noise(self, nm):
        super(Processor, type(self)).noise.fset(self, nm)
        self._source = Source.from_noise_model(nm)
        self._phase_quantization = nm.phase_imprecision
        if isinstance(self._input_state, BasicState):
            self._generate_noisy_input()

    @property
    def source_distribution(self) -> Union[SVDistribution, None]:
        r"""
        Retrieve the computed input distribution.
        :return: the input SVDistribution if `with_input` was called previously, otherwise None.
        """
        return self._inputs_map

    @property
    def source(self):
        """
        :return: The photonic source
        """
        return self._source

    @source.setter
    # When removing this method don't forget to also change the _init_noise method
    @deprecated(version="0.11.0", reason="Use noise model instead of source")
    def source(self, source: Source):
        r"""
        :param source: A Source instance to use as the new source for this processor.
        Input distribution is reset when a source is set, so `with_input` has to be called again afterwards.
        """
        self._source = source
        self._inputs_map = None

    def _init_circuit(self, m_circuit):
        if isinstance(m_circuit, ACircuit):
            self._n_moi = m_circuit.m
            self.add(0, m_circuit)
        else:
            self._n_moi = m_circuit  # number of modes of interest (MOI)

    def _init_backend(self, backend):
        if isinstance(backend, str):
            assert backend in BACKEND_LIST, f"Simulation backend '{backend}' does not exist"
            self.backend = BACKEND_LIST[backend]()
        else:
            self.backend = backend

    def type(self) -> ProcessorType:
        return ProcessorType.SIMULATOR

    @property
    def is_remote(self) -> bool:
        return False

    @dispatch(LogicalState)
    def with_input(self, input_state: LogicalState) -> None:
        r"""
        Set up the processor input with a LogicalState. Computes the input probability distribution.

        :param input_state: A LogicalState of length the input port count. Enclosed values have to match with ports
        encoding.
        """
        self._with_logical_input(input_state)

    def _generate_noisy_input(self):
        self._inputs_map = self._source.generate_distribution(self._input_state)

    @dispatch(BasicState)
    def with_input(self, input_state: BasicState) -> None:
        """
        Simulates plugging the photonic source on certain modes and turning it on.
        Computes the input probability distribution

        :param input_state: Expected input BasicState of length `self.m` (heralded modes are managed automatically)
        The properties of the source will alter the input state. A perfect source always delivers the expected state as
        an input. Imperfect ones won't.
        """
        if 'min_detected_photons' in self._parameters:
            self._min_detected_photons = self._parameters['min_detected_photons']
        if not self._min_detected_photons and self._source.is_perfect():
            self._min_detected_photons = input_state.n + list(self.heralds.values()).count(1)
        super().with_input(input_state)
        self._generate_noisy_input()

    @dispatch(StateVector)
    def with_input(self, sv: StateVector):
        r"""
        Setting directly state vector as input of a processor, use SVDistribution input

        :param sv: the state vector
        """
        return self.with_input(SVDistribution(sv))

    @dispatch(SVDistribution)
    def with_input(self, svd: SVDistribution):
        r"""
        Processor input can be set 100% manually via a state vector distribution, bypassing the source.

        :param svd: The input SVDistribution which won't be changed in any way by the source.
        Every state vector size has to be equal to `self.circuit_size`
        """
        assert self.m is not None, "A circuit has to be set before the input distribution"
        self._input_state = svd
        expected_photons = inf
        for sv in svd:
            for state in sv.keys():
                expected_photons = min(expected_photons, state.n)
                if state.m != self.circuit_size:
                    raise ValueError(
                        f'Input distribution contains states with a bad size ({state.m}), expected {self.circuit_size}')
        self._inputs_map = svd
        if 'min_detected_photons' in self._parameters:
            self._min_detected_photons = self._parameters['min_detected_photons']
        if self._min_detected_photons is None:
            self._deduce_min_detected_photons(expected_photons)

    def _circuit_changed(self):
        # Override parent's method to reset the internal simulator as soon as the component list changes
        self._simulator = None

    def with_polarized_input(self, bs: BasicState):
        assert bs.has_polarization, "BasicState is not polarized, please use with_input instead"
        self._input_state = bs
        self._inputs_map = SVDistribution(bs)
        if 'min_detected_photons' in self._parameters:
            self._min_detected_photons = self._parameters['min_detected_photons']
        if self._min_detected_photons is None:
            self._deduce_min_detected_photons(bs.n)

    def clear_input_and_circuit(self, new_m=None):
        super().clear_input_and_circuit(new_m)
        self._inputs_map = None

    def _compose_processor(self, connector, processor, keep_port: bool):
        assert isinstance(processor, Processor), "can not mix types of processors"
        super(Processor, self)._compose_processor(connector, processor, keep_port)

    def _state_preselected_physical(self, input_state: StateVector) -> bool:
        return max(input_state.n) >= self._min_detected_photons

    def _state_selected_physical(self, output_state: BasicState) -> bool:
        if self.is_threshold:
            modes_with_photons = len([n for n in output_state if n > 0])
            return modes_with_photons >= self._min_detected_photons
        return output_state.n >= self._min_detected_photons

    def linear_circuit(self, flatten: bool = False) -> Circuit:
        """
        Creates a linear circuit from internal components, if all internal components are unitary. Takes phase
        imprecision noise into account.

        :param flatten: if True, the component recursive hierarchy is discarded, making the output circuit "flat".
        :raises RuntimeError: If any component is non-unitary
        :return: The resulting Circuit object
        """
        circuit = super().linear_circuit(flatten)
        if not self._phase_quantization:
            return circuit
        # Apply phase quantization noise on all phase parameters in the circuit
        circuit = circuit.copy()  # Copy the whole circuit in order to keep the initial phase values in self
        for _, component in circuit:
            if "phi" in component.params:
                phi_param = component.param("phi")
                phi_param.set_value(self._phase_quantization * round(float(phi_param) / self._phase_quantization),
                                    force=True)
        return circuit

    def samples(self, max_samples: int, max_shots: int = None, progress_callback=None) -> Dict:
        from perceval.simulators import NoisySamplingSimulator
        assert isinstance(self.backend, ASamplingBackend), "A sampling backend is required to call samples method"
        sampling_simulator = NoisySamplingSimulator(self.backend)
        sampling_simulator.set_circuit(self.linear_circuit())
        sampling_simulator.set_selection(self._min_detected_photons, self.post_select_fn, self.heralds)
        sampling_simulator.set_threshold_detector(self.is_threshold)
        sampling_simulator.keep_heralds(False)
        return sampling_simulator.samples(self._inputs_map, max_samples, max_shots, progress_callback)

    def probs(self, precision: float = None, progress_callback: Callable = None) -> Dict:
        # assert self._inputs_map is not None, "Input is missing, please call with_inputs()"
        if self._simulator is None:
            from perceval.simulators import SimulatorFactory  # Avoids a circular import
            self._simulator = SimulatorFactory.build(self)
        else:
            self._simulator.set_circuit(self.linear_circuit() if self._is_unitary else self.components)

        if precision is not None:
            self._simulator.set_precision(precision)
        res = self._simulator.probs_svd(self._inputs_map, progress_callback=progress_callback)
        pperf = 1
        postprocessed_res = BSDistribution()
        for state, prob in res['results'].items():
            if self._state_selected_physical(state):
                postprocessed_res[self.postprocess_output(state)] += prob
            else:
                pperf -= prob

        postprocessed_res.normalize()
        res['physical_perf'] = res['physical_perf']*pperf if 'physical_perf' in res else pperf
        res['results'] = postprocessed_res
        return res

    @property
    def available_commands(self) -> List[str]:
        return ["samples" if isinstance(self.backend, ASamplingBackend) else "probs"]
