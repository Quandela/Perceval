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

import numpy as np
import sympy as sp

from .template import Backend
from perceval.utils import Matrix, BasicState
import exqalibur as xq
import time
import numba

@numba.vectorize([numba.float64(numba.complex128),numba.float32(numba.complex64)])
def _abs2(x):
    """optimized computing"""
    return x.real**2 + x.imag**2

class ComputePath:
    """A `ComputePath` is the minimal computing graph for covering a set of input states
    """

    def __init__(self, n, states, targets, backend):
        self._n = n
        self._backend = backend
        self.coefs = Matrix.zeros((backend._n_par_unitaries, backend.mk_l[n]),
                                  use_symbolic=self._backend.is_symbolic())
        if n == 0:
            self.coefs.fill(1)

        if targets is None:
            targets = [list(state) for state in states]
        self._targets = []
        self._states = []
        self._children = {}
        self.norm_coefficients = None
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
                for k in range(self._backend._n_par_unitaries):
                    self.coefs[k].fill(0)
                    for parent_idx, coef_parent in enumerate(parent_coefs):
                        for j in range(self._backend._realm):
                            idx = self._backend.fsms[self._n].get(parent_idx, j)
                            if idx != xq.npos:
                                self.coefs[k][idx] += coef_parent * u[k][j, mk]
            else:
                self._backend._compute_time += self._backend.fsms[self._n].compute_slos_layer_par(u,
                                                                   self._backend._realm,
                                                                   mk,
                                                                   self.coefs, parent_coefs,
                                                                   self._backend._n_par_unitaries,
                                                                   self._backend._parallel)/1e9

        for mk, child in self._children.items():
            child.compute(u, self.coefs, mk)


class SLOSBackend(Backend):
    """Strong Linear Optic Simulator"""
    name = "SLOS"
    supports_symbolic = True
    supports_circuit_computing = False

    def __init__(self, u, use_symbolic=None, n=None, mask=None, parallel=True):
        super().__init__(u, use_symbolic=use_symbolic, n=n, mask=mask)
        self._compute_path = None
        self._par_unitary_array = None
        self._n_par_unitaries = 0
        self._changed_unitary(None)
        self._needs_calculation = True
        self._parallel = parallel
        self._compute_time = None
        self._compile_time = 0


    def set_multi_unitary(self, list_u, single_mode=False):
        """Set a unitary list to be simulated, if single_mode is True, the unitary list is treated as a single unitary
        otherwise, each unitary is used for parallel computing - can be only used with all_prob command"""
        self._single_mode = single_mode
        prev_u = self._U
        self._U = list_u[0]
        self._n_par_unitaries = len(list_u)
        if prev_u is None or prev_u.shape != self._U.shape or self._par_unitary_array.shape[0] < len(list_u):
            self.mk_l = [1]
            self.fsms = [[]]
            self.fsas = {}
            self._compute_path = None
            self._compile_time = 0
            self.state_mapping = {}
            self._par_unitary_array = np.zeros((len(list_u), self._U.shape[0], self._U[0].shape[0]),
                                               dtype=np.complex128)
        for i, u in enumerate(list_u):
            self._par_unitary_array[i] = u
        self._needs_calculation = True

    def _changed_unitary(self, prev_u):
        u = self._U
        self._U = prev_u
        self.set_multi_unitary([u], single_mode=True)

    def _compilation(self, input_states):
        # allocate the fsas and fsms for covering all the input_states respecting possible mask
        # after calculation, we only need to keep fsa for input_state n
        # during calculation we need to keep current fsa and previous fsa
        if not self.fsas:
            current_fsa = xq.FSArray(self._realm, 0)
        else:
            current_fsa = self.fsas[max(self.fsas.keys())]
        for input_state in input_states:
            if input_state.n < len(self.fsms) and input_state.n not in self.fsas:
                # we are missing the intermediate states - let us retrieve/load it back
                current_fsa = (self._mask and xq.FSArray(self._realm, input_state.n, self._mask)
                               or xq.FSArray(self._realm, input_state.n))
            for k in range(len(self.fsms), input_state.n+1):
                fsa_n_m1 = current_fsa
                if self._mask:
                    current_fsa = xq.FSArray(self._realm, k, self._mask)
                else:
                    current_fsa = xq.FSArray(self._realm, k)
                self.mk_l.append(current_fsa.count())
                self.fsms.append(xq.FSMap(current_fsa, fsa_n_m1, True))
            if input_state.n not in self.fsas:
                self.fsas[input_state.n] = current_fsa

    def get_mask(self):
        return self._mask

    def _calculation(self):
        """
        Simulation step: update computation path coef with unitary U
        :return:
        """
        self._compute_time = 0
        self._compute_path.compute(self._par_unitary_array)
        self._needs_calculation = False

    numba.njit()
    def _build_norm_coefficients(self, input_state):
        coefs = np.zeros(self.fsas[input_state.n].count(), dtype=np.float64)
        input_state_prodnfact = input_state.prodnfact()
        # the following can be optimized to be run in parallel
        for idx, o in enumerate(self.fsas[input_state.n]):
            coefs[idx] = 1 / (o.prodnfact() * input_state_prodnfact)
        return coefs

    def compile(self, input_states):
        if isinstance(input_states, BasicState):
            input_states = [input_states]
        start = time.time()
        # build the necessary fsa/fsms
        self._compilation(input_states)
        # now check if we have a path for the input states
        found_new = False
        for input_state in input_states:
            if found_new:
                break
            found_new = input_state not in self.state_mapping
        if found_new:
            self._compute_path = ComputePath(0, input_states, None, self)
            self._needs_calculation = True
        for input_state in input_states:
            if self.state_mapping[input_state].norm_coefficients is None:
                self.state_mapping[input_state].norm_coefficients = self._build_norm_coefficients(input_state)
        self._compile_time += time.time() - start
        if self._needs_calculation:
            self._calculation()
        return True

    def probampli_be(self, input_state, output_state, norm=True):
        assert self._single_mode, "probampli cannot operate on multi-unitary mode"
        if input_state.n != output_state.n:
            return 0
        output_idx = self.fsas[output_state.n].find(output_state)
        assert output_idx != xq.npos
        non_normalized_result = self.state_mapping[input_state].coefs[0, output_idx]
        if not norm:
            return non_normalized_result
        if self._use_symbolic:
            return non_normalized_result * sp.sqrt(output_state.prodnfact()/input_state.prodnfact())
        else:
            return non_normalized_result * np.sqrt(output_state.prodnfact()/input_state.prodnfact())

    def prob_be(self, input_state, output_state):
        assert self._single_mode, "prob cannot operate on multi-unitary mode"
        return abs(self.probampli_be(input_state, output_state, False))**2\
               * output_state.prodnfact()/input_state.prodnfact()

    # The following SLOS-specific optimization is broken for polarized states/circuits
    def all_prob(self, input_state):
         self.compile(input_state)
         # precompute once for all the normalization coefficients that will be used to normalize the output states
         # probabilities
         l_c = []
         for k in range(self._n_par_unitaries):
            c = self.state_mapping[input_state].coefs[k].reshape(self.fsas[input_state.n].count())
            l_c.append(np.multiply(_abs2(c), self.state_mapping[input_state].norm_coefficients))
         if self._single_mode:
             return l_c[0]
         else:
             return l_c

    @staticmethod
    def preferred_command() -> str:
        return 'probampli'
