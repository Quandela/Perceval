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
from perceval.utils.states import BasicState

from ._abstract_backends import AStrongSimulationBackend


class SLOSCPPBackend(AStrongSimulationBackend):

    def __init__(self, mask=None):
        super().__init__()
        self._slos = xq.SLOS()
        self._fock_space = None
        if mask is not None:
            self.set_mask(mask)

    def set_circuit(self, circuit: ACircuit):
        super().set_circuit(circuit)  # Computes circuit unitary as _umat
        self._slos.set_unitary(self._umat)

    def set_input_state(self, input_state: BasicState):
        super().set_input_state(input_state)
        if self._fock_space is None or self._fock_space.m != input_state.m or self._fock_space.n != input_state.n:
            self._fock_space = xq.FSArray(input_state.m, input_state.n)

    def _init_mask(self):
        super()._init_mask()
        # self._slos.set_mask(self._mask)

    def prob_amplitude(self, output_state: FockState) -> complex:
        self._slos.set_input_state(self._input_state)
        all_pa = self._slos.all_amplitudes()
        return all_pa[self._fock_space.find(output_state)]
        # if self._mask:
        #     return all_pa[xq.FSArray(self._input_state.m, self._input_state.n, self._mask).find(output_state)]
        # else:
        #     return all_pa[xq.FSArray(self._input_state.m, self._input_state.n).find(output_state)]

    def prob_distribution(self) -> BSDistribution:
        self._slos.set_input_state(self._input_state)
        if self._mask is None:
            return self._slos.distribution()

        res = BSDistribution()
        all_probs = self._slos.all_probabilities()
        for p, fs in zip(all_probs, self._fock_space):
            if self._mask.match(fs):
                res.add(fs, p)
        return res

        # return self._slos.distribution()

    def all_prob_ampli(self):
        self._slos.set_input_state(self._input_state)
        return self._slos.all_amplitudes()

    @property
    def name(self) -> str:
        return "SLOS_CPP"

    def all_prob(self, input_state: FockState = None):
        self._slos.set_input_state(input_state or self._input_state)
        return self._slos.all_probabilities()

    def evolve(self) -> StateVector:
        self._slos.set_input_state(self._input_state)
        all_pa = self._slos.all_amplitudes()
        res = StateVector()
        for output_state, pa in zip(self._fock_space, all_pa):
            # Utterly non-optimized. Mask management should be added in the computation
            if self._mask is None or self._mask.match(output_state):
                res += output_state * pa
        # for output_state, pa in zip(self._get_iterator(self._input_state), all_pa):
        #     res += output_state * pa
        return res
