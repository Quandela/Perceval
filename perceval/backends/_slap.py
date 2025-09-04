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
import exqalibur as xq
from perceval.utils import FockState, BSDistribution, StateVector
from perceval.components import ACircuit

from ._abstract_backends import AStrongSimulationBackend


class SLAPBackend(AStrongSimulationBackend):

    def __init__(self, mask=None):
        super().__init__()
        self._slap = xq.SLAP()
        if mask is not None:
            self.set_mask(mask)

    def set_circuit(self, circuit: ACircuit):
        super().set_circuit(circuit)  # Computes circuit unitary as _umat
        self._slap.set_unitary(self._umat)

    def set_mask(self, masks: str | list[str], n = None):
        super().set_mask(masks, n)
        if self._mask:
            self._slap.set_mask(self._mask)
        else:
            self._slap.reset_mask()

    def set_input_state(self, input_state: FockState):
        super().set_input_state(input_state)
        if self._mask:
            self._slap.set_mask(self._mask)
        else:
            self._slap.reset_mask()

    def prob_amplitude(self, output_state: FockState) -> complex:
        istate = self._input_state
        all_pa = self._slap.all_prob_ampli(istate)
        if self._mask:
            return all_pa[xq.FSArray(self._input_state.m, self._input_state.n, self._mask).find(output_state)]
        else:
            return all_pa[xq.FSArray(self._input_state.m, self._input_state.n).find(output_state)]

    def prob_distribution(self) -> BSDistribution:
        istate = self._input_state
        all_probs = self._slap.all_prob(istate)

        bsd = BSDistribution()
        for output_state, probability in zip(self._get_iterator(istate), all_probs):
            bsd.add(output_state, probability)
        return bsd

    @property
    def name(self) -> str:
        return "SLAP"

    def all_prob(self, input_state: FockState = None):
        if input_state is not None:
            self.set_input_state(input_state)
        else:
            input_state = self._input_state
        return self._slap.all_prob(input_state)

    def evolve(self) -> StateVector:
        istate = self._input_state
        all_pa = self._slap.all_prob_ampli(istate)
        res = StateVector()
        for output_state, pa in zip(self._get_iterator(self._input_state), all_pa):
            res += output_state * pa
        return res
