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
from perceval.utils import BasicState, BSDistribution, StateVector
from perceval.components import ACircuit

from ._abstract_backends import AStrongSimulationBackend


class SLAPBackend(AStrongSimulationBackend):

    def __init__(self):
        super().__init__()
        self._stree = xq.SLOSTree()
        self._fock_space = None

    def set_circuit(self, circuit: ACircuit):
        super().set_circuit(circuit)  # Computes circuit unitary as _umat
        self._stree.set_unitary(self._umat)

    def set_input_state(self, input_state: BasicState):
        super().set_input_state(input_state)
        if self._fock_space is None or self._fock_space.m != input_state.m or self._fock_space.n != input_state.n:
            self._fock_space = xq.FSArray(input_state.m, input_state.n)

    def prob_amplitude(self, output_state: BasicState) -> complex:
        istate = self._input_state
        all_pa = self._stree.all_prob_ampli(istate)
        return all_pa[self._fock_space.find(output_state)]

    def prob_distribution(self) -> BSDistribution:
        istate = self._input_state
        all_probs = self._stree.all_prob(istate)

        if self._mask is not None:  # Utterly non-optimized. Mask management should be added in the computation
            all_probs = [p for p, fs in zip(all_probs, self._fock_space) if self._mask.match(fs)]

        bsd = BSDistribution()
        for output_state, probability in zip(self._get_iterator(istate), all_probs):
            bsd.add(output_state, probability)
        return bsd

    @property
    def name(self) -> str:
        return "SLAP"

    def all_prob(self, input_state: BasicState = None):
        if input_state is not None:
            self.set_input_state(input_state)
        else:
            input_state = self._input_state
        return self._stree.all_prob(input_state)

    def evolve(self) -> StateVector:
        istate = self._input_state
        all_pa = self._stree.all_prob_ampli(istate)
        res = StateVector()
        for output_state, pa in zip(self._fock_space, all_pa):
            res += output_state * pa
        res.normalize()
        return res
