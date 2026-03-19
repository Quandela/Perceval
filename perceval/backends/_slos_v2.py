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
from perceval.utils.postselect import PostSelect
from perceval.utils.states import BasicState

from ._abstract_backends import ABackend


class SLOSV2Backend(ABackend):

    def __init__(self, mask=None):
        super().__init__()
        self._slos = xq.SLOS_V2()
        if mask:
            self.set_mask(mask)
        self._post_select = None

    def set_circuit(self, circuit: ACircuit):
        super().set_circuit(circuit)  # Computes circuit unitary as _umat
        self._slos.set_unitary(self._umat)
        if self._circuit and self._input_state:
            assert self._circuit.m == self._input_state.m, f'Circuit({self._circuit.m}) and state({self._input_state}) size mismatch'

    def set_input_state(self, input_state: BasicState):
        self._input_state = input_state
        self._slos.set_input_state(input_state)
        if self._circuit and self._input_state:
            assert self._circuit.m == self._input_state.m, f'Circuit({self._circuit.m}) and state({self._input_state}) size mismatch'

    def _init_mask(self):
        super()._init_mask()
        self._slos.set_mask(self._mask)

    def set_mask(self, masks: str | list[str], n = None, at_least_modes = None):
        if isinstance(masks, str):
            masks = [masks]
        mask_length = len(masks[0])
        for mask in masks:
            mask = mask.replace("*", " ")
            assert len(mask) == mask_length, "Inconsistent mask lengths"
        fsmask = None
        if masks is not None:
            if at_least_modes:
                fsmask = xq.FSMask(mask_length, n or self._input_state.n, masks, at_least_modes)
            else:
                fsmask = xq.FSMask(mask_length, n or self._input_state.n, masks)
        self._slos.set_mask(fsmask)

    def set_post_select(self, post_selection: PostSelect):
        self._slos.set_post_select(post_selection)

    def prob_amplitude(self, output_state: FockState) -> complex:
        all_pa = self._slos.all_amplitudes()
        idx = self._slos.get_index(output_state)
        return all_pa[idx]

    def probability(self, output_state: FockState) -> float:
        return abs(self.prob_amplitude(output_state)) ** 2

    def prob_distribution(self) -> BSDistribution:
        return self._slos.distribution()

    def all_prob_ampli(self):
        return self._slos.all_amplitudes()

    @property
    def name(self) -> str:
        return "SLOS_V2"

    def all_prob(self, input_state: FockState = None):
        return self._slos.all_probabilities()

    def evolve(self) -> StateVector:
        self._slos.set_input_state(self._input_state)
        all_pa = self._slos.all_amplitudes()
        res = StateVector()
        for output_state, pa in zip(self._slos.get_states(), all_pa):
            res += output_state * pa
        return res
