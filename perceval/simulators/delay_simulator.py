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

from .simulator_interface import ASimulatorDecorator
from ._simulator_utils import _retrieve_mode_count, _unitary_components_to_circuit
from perceval.components import ACircuit, PERM, TD
from perceval.utils import BasicState, global_params

from enum import Enum
from typing import List


class _CType(Enum):
    UNITARY = 0
    DELAY = 1
    OTHER = 2


def _count_total_delay(component_list: List) -> int:
    return int(sum([c.get_variables()["t"] if isinstance(c, TD) else 0 for _, c in component_list]))

def _compute_depth(component_list: List, mode_count: int) -> int:
    depth = 1
    count_per_mode = [[0, {i}] for i in range(mode_count)]
    for r, c in component_list:
        if isinstance(c, TD):
            t = c.get_variables()["t"]
            assert not isinstance(t, str), "Delay parameter %s not set" % t
            if t != int(t):
                raise NotImplementedError("Only round time delays are supported")
            t = int(t)
            r = r[0]
            if not len(count_per_mode[r][1]) == 1:
                cur_count = max(count_per_mode[i][0] for i in count_per_mode[r][1])
                depth += cur_count
                for mode in count_per_mode[r][1]:
                    count_per_mode[mode][1] = {mode}
                for i in range(mode_count):
                    count_per_mode[i][0] = max(0, count_per_mode[i][0] - cur_count)
            count_per_mode[r][0] += t

        elif len(r) > 1:
            set_r = set(r)
            for mode in r:
                count_per_mode[mode][1] |= set_r

    depth += max(count_per_mode[i][0] for i in range(mode_count))
    return depth


class DelaySimulator(ASimulatorDecorator):
    def __init__(self, simulator):
        super().__init__(simulator)
        self._original_m: int = 0
        self._expanded_m: int = 0
        self._depth: int = 0

    def _prepare_input(self, input_state):
        expanded_input = input_state ** self._depth
        return expanded_input * BasicState([0] * (self._expanded_m - self._depth*self._original_m))

    def _prepare_circuit(self, circuit):
        self._original_m = _retrieve_mode_count(circuit)
        expanded_circuit, expanded_mode_count = self._expand_td(circuit)
        self._expanded_m = expanded_mode_count
        return expanded_circuit

    def _postprocess_results(self, results):
        output = type(results)()
        mode_range = [(self._depth - 1) * self._original_m, self._depth * self._original_m]
        for out_state, output_prob in results.items():
            if output_prob > global_params['min_p']:
                reduced_out_state = out_state[mode_range[0]:mode_range[1]]
                output[reduced_out_state] += output_prob
        return output

    def _expand_td(self, component_list: List):
        mode_count = self._original_m
        expanded = []
        current_chunk = []
        can_output_circuit = True
        for r, c in component_list:
            if isinstance(c, ACircuit):  # case unitary component
                current_chunk.append([r, c])

            elif not isinstance(c, TD):  # case other non-unitary component
                expanded.append([_CType.UNITARY, current_chunk.copy()])
                current_chunk = []
                expanded.append([_CType.OTHER, [[r, c]]])
                can_output_circuit = False

            else:  # case time delay
                t = int(c.get_variables()["t"])
                r0 = r[0]

                if r0 != mode_count - 1:
                    # Add a fake permutation to put the TD at the last mode
                    r = list(range(r0, mode_count))
                    perm_list = [mode_count - r0 - 1] + list(range(1, mode_count - r0 - 1)) + [0]
                    current_chunk.append([r, PERM(perm_list)])

                for _ in range(t):
                    expanded.append([_CType.UNITARY, current_chunk.copy()])
                    expanded.append([_CType.DELAY, []])
                    current_chunk = []

                if r0 != mode_count - 1:
                    # Nullify the fake permutation
                    current_chunk.append([r, PERM(perm_list)])

        expanded.append([_CType.UNITARY, current_chunk.copy()])
        self._depth = _compute_depth(component_list, mode_count)
        total_delay = _count_total_delay(component_list)
        new_m = self._depth * mode_count + total_delay
        new_circ = []

        for d in range(self._depth):
            i_td = 0
            for i, type_and_cur_U in enumerate(expanded):
                ctype, current = type_and_cur_U
                if ctype != _CType.DELAY:
                    for r, c in current:
                        new_circ.append((r[0] + d * mode_count, c))

                # Then we permute to mimic TD
                else:
                    r0 = (d + 1) * mode_count - 1
                    perm_list = [new_m - i_td - r0 - 1] + list(range(1, new_m - i_td - r0 - 1)) + [0]
                    i_td += 1
                    new_circ.append((r0, PERM(perm_list)))

        if can_output_circuit:
            new_circ = _unitary_components_to_circuit(new_circ, new_m)
        return new_circ, new_m
