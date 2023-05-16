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
from perceval.components import ACircuit, LC, PERM, BS
from perceval.utils import BasicState

from typing import List


class LossSimulator(ASimulatorDecorator):

    def _prepare_input(self, input_state):
        return input_state * BasicState([0] * (self._expanded_m - self._original_m))

    def _prepare_circuit(self, circuit):
        self._original_m = _retrieve_mode_count(circuit)
        expanded_circuit = self._simulate_losses_with_beam_splitters(circuit)
        return expanded_circuit

    def _postprocess_results(self, results):
        output = type(results)()
        for out_state, output_prob in results.items():
            reduced_out_state = out_state[0:self._original_m]
            output[reduced_out_state] += output_prob
        return output

    def _simulate_losses_with_beam_splitters(self, components: List) -> ACircuit:
        output = []
        can_output_circuit = True
        next_free_mode = self._original_m
        for r, c in components:
            if isinstance(c, ACircuit):  # case unitary component
                output.append((r, c))

            elif not isinstance(c, LC):  # case other non-unitary component
                output.append((r, c))
                can_output_circuit = False

            else:  # case loss channel
                if r[0] != next_free_mode - 1:
                    r_ip = tuple(range(r[0]+1, next_free_mode+1))
                    in_perm = [next_free_mode-r[0]-1] + [m for m in range(1, next_free_mode-r[0]-1)] + [0]
                    output.append((r_ip, PERM(in_perm)))

                loss = c.get_variables()["loss"]
                bs = BS.H(BS.r_to_theta(1 - loss))
                r_bs = (r[0], r[0]+1)
                output.append((r_bs, bs))

                if r[0] != next_free_mode - 1:
                    out_perm = PERM(in_perm)
                    out_perm.inverse(h=True)
                    output.append((r_ip, out_perm))
                next_free_mode += 1

        self._expanded_m = next_free_mode
        if can_output_circuit:
            output = _unitary_components_to_circuit(output, self._expanded_m)

        return output
