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

from perceval.backends import SLOSBackend, ASamplingBackend
from perceval.components import Circuit, ACircuit
from perceval.utils import FockState

class StepperBackend(ASamplingBackend):
    def __init__(self):
        super().__init__()
        self._backend = SLOSBackend()

    def set_circuit(self, circuit: ACircuit):
        if isinstance(circuit, Circuit):
            self._circuit = circuit
        else:
            self._circuit = Circuit(circuit.m).add(0, circuit)

    def set_input_state(self, input_state: FockState):
        super().set_input_state(input_state)

    def sample(self):
        m = self._circuit.m
        current_circuit = Circuit(m)
        current_state = self._input_state
        for r, c in self._circuit:
            current_circuit.add(r, c)
            self._backend.set_circuit(current_circuit)
            self._backend.set_mask(''.join([ ' ' if i in r else chr(ord('0') + current_state[i]) for i in range(0, m) ]))
            self._backend.set_input_state(self._input_state)
            current_state = self._backend.prob_distribution().sample(1)[0]

        return current_state

    def samples(self, count: int):
        """
        Request `count` samples from the circuit given an input state.
        """
        return [self.sample() for _ in range(count)]

    @property
    def name(self) -> str:
        return "StepperSampler"
