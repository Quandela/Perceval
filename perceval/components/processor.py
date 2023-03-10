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
from .computation import count_TD, count_independant_TD, expand_TD
from perceval.utils import SVDistribution, BSDistribution, BSSamples, BasicState, StateVector, global_params
from perceval.backends import BACKEND_LIST
from perceval.backends.processor import StepperBackend

from multipledispatch import dispatch
from typing import Dict, Callable, Union, List


class Processor(AProcessor):
    """
    Generic definition of processor as a source + components (both unitary and non-unitary) + ports
    + optional post-processing logic

    :param backend_name: Name of the simulator backend to run
    :param m_circuit: can either be:

        - an int: number of modes of interest (MOI). A mode of interest is any non-heralded mode.
        >>> p = Processor("SLOS", 5)

        - a circuit: the input circuit to start with. Other components can still be added afterwards with `add()`
        >>> p = Processor("SLOS", BS() // PS() // BS())

    :param source: the Source used by the processor (defaults to perfect source)
    """
    def __init__(self, backend_name: str, m_circuit: Union[int, ACircuit] = None, source: Source = Source(),
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
        self._simulator = None
        assert backend_name in BACKEND_LIST, f"Simulation backend '{backend_name}' does not exist"
        self._backend_name = backend_name

    def _setup_simulator(self, **kwargs):
        if self._is_unitary:
            self._simulator = BACKEND_LIST[self._backend_name](self.linear_circuit(), **kwargs)
        else:
            if "probampli" not in BACKEND_LIST[self._backend_name].available_commands():
                raise RuntimeError(f"{self._backend_name} backend cannot be used on a non-unitary processor")
            self._simulator = StepperBackend(self.non_unitary_circuit(),
                                             m=self.circuit_size,
                                             backend_name=self._backend_name,
                                             min_detected_photons_filter=self._min_detected_photons)

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

    @dispatch(BasicState)
    def with_input(self, input_state: BasicState) -> None:
        """
        Simulates plugging the photonic source on certain modes and turning it on.
        Computes the input probability distribution

        :param input_state: Expected input BasicState of length `self.m` (heralded modes are managed
        automatically)
        The properties of the source will alter the input state. A perfect source always delivers the expected state as
        an input. Imperfect ones won't.
        """
        input_list = [0] * self.circuit_size
        self._inputs_map = SVDistribution()
        assert self.m is not None, "A circuit has to be set before the input state"
        expected_input_length = self.m
        assert len(input_state) == expected_input_length, \
            f"Input length not compatible with circuit (expects {expected_input_length}, got {len(input_state)})"
        input_idx = 0
        expected_photons = 0
        for k in range(self.circuit_size):
            distribution = SVDistribution(StateVector("|0>"))
            if k in self.heralds:
                if self.heralds[k] == 1:
                    distribution = self._source.probability_distribution()
                    input_list[k] = 1
                    expected_photons += 1
            else:
                if input_state[input_idx] > 0:
                    distribution = self._source.probability_distribution(input_state[input_idx])
                    input_list[k] = input_state[input_idx]
                    expected_photons += 1
                input_idx += 1
            self._inputs_map *= distribution  # combine distributions

        # Needed to do this at the end
        used_input_map = SVDistribution()
        for state, prob in self._inputs_map.items():
            if prob:
                used_input_map[_find_equivalent(state[0])] += prob

        self._inputs_map = used_input_map

        self._input_state = BasicState(input_list)
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

    def clear_input_and_circuit(self):
        super().clear_input_and_circuit()
        self._inputs_map = None
        self._simulator = None

    def _compose_processor(self, connector, processor, keep_port: bool):
        assert isinstance(processor, Processor), "can not mix types of processors"
        super(Processor, self)._compose_processor(connector, processor, keep_port)

    def _init_command(self, command_name: str):
        assert self._inputs_map is not None, "Input is missing, please call with_inputs()"
        if self._backend_name == "CliffordClifford2017" and self._has_td:
            raise NotImplementedError(
                "Time delays are not implemented within CliffordClifford2017 backend. Please use another one.")
        if not self._has_td:  # TODO: remove quickfix by something clever :  self._simulator is None and
            self._setup_simulator()

    def _state_preselected_physical(self, input_state: StateVector) -> bool:
        return max(input_state.n) >= self._min_detected_photons

    def _state_selected_physical(self, output_state: BasicState) -> bool:
        if self.is_threshold:
            modes_with_photons = len([n for n in output_state if n > 0])
            return modes_with_photons >= self._min_detected_photons
        return output_state.n >= self._min_detected_photons

    def samples(self, count: int, progress_callback=None) -> Dict:
        self._init_command("samples")
        output = BSSamples()
        not_selected_physical = 0
        not_selected = 0
        selected_inputs = self._inputs_map.sample(count)
        idx = 0
        while len(output) < count:
            selected_input = selected_inputs[idx]
            idx += 1
            if idx == len(selected_inputs):
                idx = 0
                selected_inputs = self._inputs_map.sample(count)
            if not self._state_preselected_physical(selected_input):
                not_selected_physical += 1
                continue
            sampled_state = self._simulator.sample(selected_input)
            if not self._state_selected_physical(sampled_state):
                not_selected_physical += 1
                continue
            if self._state_selected(sampled_state):
                output.append(self.postprocess_output(sampled_state))
            else:
                not_selected += 1
            if progress_callback:
                exec_request = progress_callback(len(output)/count, "sampling")
                if exec_request is not None and 'cancel_requested' in exec_request and exec_request['cancel_requested']:
                    break

        physical_perf = (count + not_selected) / (count + not_selected + not_selected_physical)
        logical_perf = count / (count + not_selected)
        return {'results': output, 'physical_perf': physical_perf, 'logical_perf': logical_perf}

    def probs(self, progress_callback: Callable = None) -> Dict:
        self._init_command("probs")
        output = BSDistribution()
        p_logic_discard = 0
        if not self._has_td:
            input_length = len(self._inputs_map)
            physical_perf = 1

            for idx, (input_state, input_prob) in enumerate(self._inputs_map.items()):
                if not self._state_preselected_physical(input_state):
                    physical_perf -= input_prob
                else:
                    for (output_state, p) in self._simulator.allstateprob_iterator(input_state):
                        if p < global_params['min_p']:
                            continue
                        output_prob = p * input_prob
                        if not self._state_selected_physical(output_state):
                            physical_perf -= output_prob
                            continue
                        if self._state_selected(output_state):
                            output[self.postprocess_output(output_state)] += output_prob
                        else:
                            p_logic_discard += output_prob
                if progress_callback:
                    exec_request = progress_callback(idx/input_length, 'probs')
                    if exec_request is not None and 'cancel_requested' in exec_request and exec_request['cancel_requested']:
                        raise RuntimeError("Cancel requested")

        else:
            # Create a bigger processor with no heralds to represent the time delays
            p_comp = self.flatten()
            TD_number = count_TD(p_comp)
            depth = count_independant_TD(p_comp, self.circuit_size) + 1
            p_comp, extend_m = expand_TD(p_comp, depth, self.circuit_size, TD_number, True)
            # p_comp = simplify(p_comp, extend_m)
            extended_p = _expand_TD_processor(p_comp,
                                              self._backend_name,
                                              depth,
                                              extend_m,
                                              self._input_state,
                                              self._min_detected_photons,
                                              self.source)

            res = extended_p.probs(progress_callback=progress_callback)

            # Now reduce the states.
            interest_m = [(depth - 1) * self.circuit_size, depth * self.circuit_size]
            extended_out = res["results"]

            second_perf = 1
            for out_state, output_prob in extended_out.items():
                reduced_out_state = out_state[interest_m[0]: interest_m[1]]
                if not self._state_selected_physical(reduced_out_state):
                    second_perf -= output_prob
                    continue
                if self._state_selected(reduced_out_state):
                    output[self.postprocess_output(reduced_out_state)] += output_prob
                else:
                    p_logic_discard += output_prob
            physical_perf = second_perf * res["physical_perf"]

        if physical_perf < global_params['min_p']:
            physical_perf = 0
        all_p = sum(v for v in output.values())
        if all_p == 0:
            return {'results': output, 'physical_perf': physical_perf}
        logical_perf = 1 - p_logic_discard / (p_logic_discard + all_p)
        output.normalize()
        return {'results': output, 'physical_perf': physical_perf, 'logical_perf': logical_perf}

    @property
    def available_commands(self) -> List[str]:
        return [BACKEND_LIST[self._backend_name].preferred_command()=="samples" and "samples" or "probs"]


def _expand_TD_processor(components: list, backend_name: str, depth: int, m: int,
                         input_states: Union[SVDistribution, BasicState], min_detected_photons: int,
                         source: Source):
    p = Processor(backend_name, m, source)
    if isinstance(input_states, SVDistribution):
        input_states = input_states ** depth * SVDistribution(BasicState([0] * (m - depth * next(iter(input_states)).m)))
    else:  # BasicState
        input_states = input_states ** depth * BasicState([0] * (m - depth * input_states.m))

    p.with_input(input_states)
    for r, c in components:
        p.add(r, c)
    p.min_detected_photons_filter(min_detected_photons)
    return p


def _find_equivalent(state):
    if not state.has_annotations:
        return state
    annot_numbers = dict()
    i = 0
    new_state = "|"

    for mode in range(state.m):
        if not state[mode]:
            new_state += "0,"
            continue
        annotations = state.get_mode_annotations(mode)
        for n in range(state[mode]):
            annot = annotations[n]["_"]
            if annot is not None:
                nb = int(annot.real)
                if nb not in annot_numbers:
                    annot_numbers[nb] = i
                    i += 1
                new_state += "{_:" + f"{annot_numbers[nb]}" + "}"
        new_state += ","
    new_state = new_state[:-1] + ">"
    return StateVector(new_state)
