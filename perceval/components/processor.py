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
from numpy import Inf

from .abstract_processor import AProcessor, ProcessorType
from .port import LogicalState
from .source import Source
from .linear_circuit import ACircuit
from perceval.utils import SVDistribution, BSDistribution, BSSamples, BasicState, StateVector
from perceval.backends import ABackend, ASamplingBackend, BACKEND_LIST

from multipledispatch import dispatch
from typing import Dict, Callable, Union, List


class Processor(AProcessor):
    """
    Generic definition of processor as a source + components (both unitary and non-unitary) + ports
    + optional post-processing logic

    :param backend: Name or instance of a simulation backend
    :param m_circuit: can either be:

        - an int: number of modes of interest (MOI). A mode of interest is any non-heralded mode.
        >>> p = Processor("SLOS", 5)

        - a circuit: the input circuit to start with. Other components can still be added afterwards with `add()`
        >>> p = Processor("SLOS", BS() // PS() // BS())

    :param source: the Source used by the processor (defaults to perfect source)
    :param name: a textual name for the processor (defaults to "Local processor")
    """
    def __init__(self, backend: Union[ABackend, str], m_circuit: Union[int, ACircuit] = None, source: Source = Source(),
                 name: str = None):
        super().__init__()
        self._source = source
        self.name = "Local processor" if name is None else name

        if isinstance(m_circuit, ACircuit):
            self._n_moi = m_circuit.m
            self.add(0, m_circuit)
        else:
            self._n_moi = m_circuit  # number of modes of interest (MOI)

        self._inputs_map: Union[SVDistribution, None] = None
        if isinstance(backend, str):
            assert backend in BACKEND_LIST, f"Simulation backend '{backend}' does not exist"
            self.backend = BACKEND_LIST[backend]()
        else:
            self.backend = backend
        self._simulator = None

    def type(self) -> ProcessorType:
        return ProcessorType.SIMULATOR

    @property
    def is_remote(self) -> bool:
        return False

    def check_input(self, input_state: BasicState):
        assert self.m is not None, "A circuit has to be set before the input state"
        expected_input_length = self.m
        assert len(input_state) == expected_input_length, \
            f"Input length not compatible with circuit (expects {expected_input_length}, got {len(input_state)})"

    @dispatch(LogicalState)
    def with_input(self, input_state: LogicalState) -> None:
        r"""
        Set up the processor input with a LogicalState. Computes the input probability distribution.

        :param input_state: A LogicalState of length the input port count. Enclosed values have to match with ports
        encoding.
        """
        self._with_logical_input(input_state)

    @dispatch(BasicState)
    def with_input(self, input_state: BasicState) -> None:
        """
        Simulates plugging the photonic source on certain modes and turning it on.
        Computes the input probability distribution

        :param input_state: Expected input BasicState of length `self.m` (heralded modes are managed automatically)
        The properties of the source will alter the input state. A perfect source always delivers the expected state as
        an input. Imperfect ones won't.
        """
        self.check_input(input_state)
        input_list = [0] * self.circuit_size
        input_idx = 0
        expected_photons = 0
        # Build real input state (merging ancillas + expected input) and compute expected photon count
        for k in range(self.circuit_size):
            if k in self.heralds:
                input_list[k] = self.heralds[k]
                expected_photons += self.heralds[k]
            else:
                input_list[k] = input_state[input_idx]
                expected_photons += input_state[input_idx]
                input_idx += 1

        self._input_state = BasicState(input_list)
        self._inputs_map = self._source.generate_distribution(self._input_state)
        self._min_detected_photons = expected_photons
        if 'min_detected_photons' in self._parameters:
            self._min_detected_photons = self._parameters['min_detected_photons']

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
        expected_photons = Inf
        for sv in svd:
            for state in sv:
                expected_photons = min(expected_photons, state.n)
                if state.m != self.circuit_size:
                    raise ValueError(
                        f'Input distribution contains states with a bad size ({state.m}), expected {self.circuit_size}')
        self._inputs_map = svd
        self._min_detected_photons = expected_photons
        if 'min_detected_photons' in self._parameters:
            self._min_detected_photons = self._parameters['min_detected_photons']

    def _circuit_changed(self):
        # Override parent's method to reset the internal simulator as soon as the component list changes
        self._simulator = None

    def with_polarized_input(self, bs: BasicState):
        assert bs.has_polarization, "BasicState is not polarized, please use with_input instead"
        self._input_state = bs
        self._inputs_map = SVDistribution(bs)
        self._min_detected_photons = bs.n
        if 'min_detected_photons' in self._parameters:
            self._min_detected_photons = self._parameters['min_detected_photons']

    def clear_input_and_circuit(self):
        super().clear_input_and_circuit()
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

    def samples(self, count: int, progress_callback=None) -> Dict:
        assert isinstance(self.backend, ASamplingBackend), "A sampling backend is required to call samples method"
        pre_physical_perf = 1
        # Rework input map so that it contains only states with enough photons
        input_svd = SVDistribution()
        for sv, p in self._inputs_map.items():
            if self._state_preselected_physical(sv):
                if len(sv) > 1:
                    raise RuntimeError("Cannot sample on a superposed state")
                input_svd[sv] = p
            else:
                pre_physical_perf -= p

        self.backend.set_circuit(self.linear_circuit())
        output = BSSamples()
        selected_inputs = []
        idx = 0
        not_selected_physical = 0
        not_selected = 0
        while len(output) < count:
            if idx == len(selected_inputs):
                idx = 0
                selected_inputs = input_svd.sample(count)
            selected_bs = selected_inputs[idx][0]
            idx += 1

            # Sampling
            if selected_bs.has_annotations:  # In case of annotations, input must be separately sampled, then recombined
                bs_list = selected_bs.separate_state()
                sampled_components = []
                for bs in bs_list:
                    self.backend.set_input_state(bs)
                    sampled_components.append(self.backend.sample())
                sampled_state = sampled_components.pop()
                for component in sampled_components:
                    sampled_state = sampled_state.merge(component)
            else:
                self.backend.set_input_state(selected_bs)
                sampled_state = self.backend.sample()

            # Post-processing
            if not self._state_selected_physical(sampled_state):
                not_selected_physical += 1
                continue
            if self._state_selected(sampled_state):
                output.append(self.postprocess_output(sampled_state))
            else:
                not_selected += 1

            # Progress handling
            if progress_callback:
                exec_request = progress_callback(len(output)/count, "sampling")
                if exec_request is not None and 'cancel_requested' in exec_request and exec_request['cancel_requested']:
                    break

        physical_perf = pre_physical_perf * (count + not_selected) / (count + not_selected + not_selected_physical)
        logical_perf = count / (count + not_selected)
        return {'results': output, 'physical_perf': physical_perf, 'logical_perf': logical_perf}

    def probs(self, progress_callback: Callable = None) -> Dict:
        # assert self._inputs_map is not None, "Input is missing, please call with_inputs()"
        if self._simulator is None:
            from perceval.simulators import SimulatorFactory  # Avoids a circular import
            self._simulator = SimulatorFactory.build(self)
        else:
            self._simulator.set_circuit(self.linear_circuit() if self._is_unitary else self.components)

        res = self._simulator.probs_svd(self._inputs_map, progress_callback=progress_callback)
        lperf = 1
        pperf = 1
        postprocessed_res = BSDistribution()
        for state, prob in res['results'].items():
            if not self._state_selected_physical(state):
                pperf -= prob
                continue
            if self._state_selected(state):
                postprocessed_res[self.postprocess_output(state)] += prob
            else:
                lperf -= prob
        postprocessed_res.normalize()
        res['logical_perf'] = res['logical_perf']*lperf if 'logical_perf' in res else lperf
        res['physical_perf'] = res['physical_perf']*pperf if 'physical_perf' in res else pperf
        res['results'] = postprocessed_res
        return res

    @property
    def available_commands(self) -> List[str]:
        return ["samples" if self.backend.preferred_command() == "sample" else "probs"]
