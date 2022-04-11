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
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import sympy as sp

from .template import Backend
from perceval.utils import Matrix, BasicState
import quandelibc as qc


class ComputePath:
    """A `ComputePath` is the minimal computing graph for covering a set of input states
    """

    def __init__(self, n, states, targets, backend):
        self._n = n
        self._backend = backend
        self.coefs = Matrix.zeros((backend.mk_l[n], 1),
                                  use_symbolic=self._backend.is_symbolic())
        if n == 0:
            self.coefs.fill(1)

        if targets is None:
            targets = [list(state) for state in states]
        self._targets = []
        self._states = []
        self._children = {}
        for t, s in zip(targets, states):
            if sum(t) == 0:
                backend.state_mapping[s] = self
            else:
                self._targets.append(t)
                self._states.append(s)
        self._decompose()

    def _decompose(self):
        targets = self._targets
        states = self._states
        while targets:
            counts = [0] * len(targets[0])
            for one_target in targets:
                counts = [x + y for (x, y) in zip(counts, one_target)]
            max_mode = max(counts)
            max_index = counts.index(max_mode)
            current_targets = []
            current_states = []
            new_targets = []
            new_states = []
            for one_target, one_state in zip(targets, states):
                if one_target[max_index]:
                    one_target[max_index] -= 1
                    current_targets.append(one_target)
                    current_states.append(one_state)
                else:
                    new_targets.append(one_target)
                    new_states.append(one_state)
            self._children[max_index] = ComputePath(self._n+1,
                                                    current_states, current_targets,
                                                    self._backend)
            targets = new_targets
            states = new_states

    def compute(self, u, parent_coefs: Matrix = None, mk: int = None):
        r"""Given the precompiled compute path, update all the coefficients"""
        if parent_coefs is not None:
            if self._backend._use_symbolic:
                self.coefs.fill(0)
                for parent_idx, coef_parent in enumerate(parent_coefs):
                    for j in range(self._backend._realm):
                        idx = self._backend.fsms[self._n].get(parent_idx, j)
                        if idx != qc.npos:
                            self.coefs[idx] += coef_parent * u[j, mk]
            else:
                self._backend.fsms[self._n].compute_slos_layer(u, self._backend._realm, mk, self.coefs, parent_coefs)

        for mk, child in self._children.items():
            child.compute(u, self.coefs, mk)


class SLOSBackend(Backend):
    """Strong Linear Optic Simulator"""
    name = "SLOS"
    supports_symbolic = True
    supports_circuit_computing = False

    def __init__(self, u, use_symbolic=None, n=None, mask=None):
        super().__init__(u, use_symbolic=use_symbolic, n=n, mask=mask)
        self._compute_path = None
        self._changed_unitary(None)

    def _changed_unitary(self, prev_u):
        if self._compute_path is not None and prev_u is not None and prev_u.shape == self._U.shape:
            self._calculation()
        else:
            self.mk_l = [1]
            self.fsms = [[]]
            self.fsas = {}
            self._compute_path = None
            self.state_mapping = {}

    def _compilation(self, input_states):
        # allocate the fsas and fsms for covering all the input_states respecting possible mask
        # after calculation, we only need to keep fsa for input_state n
        # during calculation we need to keep current fsa and previous fsa
        if not self.fsas:
            current_fsa = qc.FSArray(self._realm, 0)
        else:
            current_fsa = self.fsas[max(self.fsas.keys())]
        for input_state in input_states:
            if input_state.n < len(self.fsms) and input_state.n not in self.fsas:
                # we are missing the intermediate states - let us retrieve/load it back
                current_fsa = (self._mask and qc.FSArray(self._realm, input_state.n, self._mask)
                               or qc.FSArray(self._realm, input_state.n))
            for k in range(len(self.fsms), input_state.n+1):
                fsa_n_m1 = current_fsa
                if self._mask:
                    current_fsa = qc.FSArray(self._realm, k, self._mask)
                else:
                    current_fsa = qc.FSArray(self._realm, k)
                self.mk_l.append(current_fsa.count())
                self.fsms.append(qc.FSMap(current_fsa, fsa_n_m1, True))
            if input_state.n not in self.fsas:
                self.fsas[input_state.n] = current_fsa

    def get_mask(self):
        return self._mask

    def _calculation(self):
        """
        Simulation step: update computation path coef with unitary U
        :return:
        """
        self._compute_path.compute(self._U)

    def compile(self, input_states):
        if isinstance(input_states, BasicState):
            input_states = [input_states]
        # build the necessary fsa/fsms
        self._compilation(input_states)
        # now check if we have a path for the input states
        found_new = False
        for input_state in input_states:
            if found_new:
                break
            found_new = input_state not in self.state_mapping
        if not found_new:
            return False
        self._compute_path = ComputePath(0, input_states, None, self)
        self._calculation()
        return True

    def probampli_be(self, input_state, output_state, n=None, output_idx=None, norm=True):
        if input_state.n != output_state.n:
            return 0
        if output_idx is None:
            output_idx = self.fsas[output_state.n].find(output_state)
            assert output_idx != qc.npos
        if not norm:
            return self.state_mapping[input_state].coefs[output_idx, 0]
        if self._use_symbolic:
            return self.state_mapping[input_state].coefs[output_idx, 0]\
                   * sp.sqrt(output_state.prodnfact()/input_state.prodnfact())
        else:
            return self.state_mapping[input_state].coefs[output_idx, 0]\
                   * np.sqrt(output_state.prodnfact()/input_state.prodnfact())

    def prob_be(self, input_state, output_state, n=None, output_idx=None):
        return abs(self.probampli_be(input_state, output_state, n, output_idx, False))**2\
               * output_state.prodnfact()/input_state.prodnfact()

    def all_prob(self, input_state):
        self.compile(input_state)
        c = np.copy(self.state_mapping[input_state].coefs).reshape(self.fsas[input_state.n].count())
        self.fsas[input_state.n].norm_coefs(c)
        c /= input_state.prodnfact()
        return abs(c)**2
