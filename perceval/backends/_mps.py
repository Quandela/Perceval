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

from typing import Union
import copy

import numpy as np
from math import factorial
from scipy.special import comb
from collections import defaultdict

from ._abstract_backends import AProbAmpliBackend
from perceval.utils import BasicState
from perceval.components import ACircuit


class MPSBackend(AProbAmpliBackend):
    """Step-by-step circuit propagation algorithm, works on a circuit.
    Approximate the probability amplitudes with a cutoff.
    - For now only supports Phase shifters and Beam Splitters
    - TODO: link to the quandelibc computation
    """

    def __init__(self):
        super().__init__()
        self._s_min = 1e-8
        self._cutoff = None
        self._compiled_input = None
        self._res = defaultdict(lambda: defaultdict(lambda: np.array([0])))
        # Doubts : Nested dictionary why?
        self._current_input = None

    @property
    def name(self) -> str:
        return "MPS"

    def set_cutoff(self, cutoff_val: int):
        assert isinstance(cutoff_val, int), "cutoff must be an integer"
        self._cutoff = cutoff_val

    def set_circuit(self, circuit: ACircuit):
        super().set_circuit(circuit)
        C = self._circuit
        for r, c in C:
            assert c.compute_unitary(use_symbolic=False).shape[0] <= 2, \
                "MPS backend can not be used with components of using more than 2 modes"
        if self._cutoff is None:
            self._cutoff = C.m

    def set_input_state(self, input_state: BasicState):
        super().set_input_state(input_state)
        self.compile()

    def apply(self, r, c):
        u = c.compute_unitary(False)
        k_mode = r[0]
        if len(u) == 2:
            self.update_state(k_mode, u)  # --> quandelibc
        elif len(u) == 1:
            self.update_state_1_mode(k_mode, u)  # --> quandelibc

    def compile(self) -> bool:
        C = self._circuit
        var = [float(p) for p in C.get_parameters()]
        if self._compiled_input and self._compiled_input[0] == var and self._input_state in self._res:
            return False
        self._compiled_input = copy.copy((var, self._input_state))
        self._current_input = None

        # TODO : allow any StateVector as in stepper, or a list as in SLOS
        # self._input_state *= BasicState([0] * (self._input_state.m - self._input_state.m))
        self.n = self._input_state.n
        self.d = self.n + 1
        self._cutoff = min(self._cutoff, self.d ** (self._input_state.m//2))
        self._gamma = np.zeros((self._input_state.m, self._cutoff, self._cutoff, self.d), dtype='complex_')
        for i in range(self._input_state.m):
            self._gamma[i, 0, 0, self._input_state[i]] = 1
        self.sv = np.zeros((self._input_state.m, self._cutoff))
        self.sv[:, 0] = 1

        for r, c in C:
            self.apply(r, c)

        self._res[tuple(self._input_state)]["gamma"] = self._gamma.copy()
        self._res[tuple(self._input_state)]["sv"] = self.sv.copy()
        return True

    def prob_amplitude(self, output_state: BasicState) -> complex:
        # TODO: put in quandelibc
        m = self._input_state.m
        mps_in_list = []
        self._current_input = tuple(self._input_state)
        for k in range(m - 1):
            mps_in_list.append(self._res[tuple(self._input_state)]["gamma"][k, :, :, output_state[k]])
            mps_in_list.append(self._sv_diag(k))
        mps_in_list.append(self._res[tuple(self._input_state)]["gamma"][self._input_state.m-1, :, :, output_state[self._input_state.m-1]])
        return np.linalg.multi_dot(mps_in_list)[0, 0]

    @staticmethod
    def preferred_command() -> str:
        return 'probampli'

# ################ From here, everything must be in quandelibc ##############################

    def _transition_matrix_1_mode(self, u):
        d = self.d

        big_u = np.zeros((d, d), dtype='complex_')
        for i in range(d):
            big_u[i, i] = u[0, 0] ** i

        return big_u

    def update_state_1_mode(self, k, u):
        self._gamma[k] = np.tensordot(self._gamma[k], self._transition_matrix_1_mode(u), axes=(2, 0))

    def update_state(self, k, u):

        if 0 < k < self._input_state.m - 2:
            theta = np.tensordot(self._sv_diag(k - 1), self._gamma[k, :], axes=(1, 0))
            theta = np.tensordot(theta, self._sv_diag(k), axes=(1, 0))
            theta = np.tensordot(theta, self._gamma[k + 1, :], axes=(2, 0))
            theta = np.tensordot(theta, self._sv_diag(k + 1), axes=(2, 0))
            theta = np.tensordot(theta, self._transition_matrix(u), axes=([1, 2], [0, 1]))
            theta = theta.swapaxes(1, 2).swapaxes(0, 1).swapaxes(2, 3)
            theta = theta.reshape(self.d * self._cutoff, self.d * self._cutoff)

        elif k == 0:
            theta = np.tensordot(self._gamma[k, :], self._sv_diag(k), axes=(1, 0))
            theta = np.tensordot(theta, self._gamma[k + 1, :], axes=(2, 0))
            theta = np.tensordot(theta, self._sv_diag(k + 1), axes=(2, 0))
            theta = np.tensordot(theta, self._transition_matrix(u),
                                 axes=([1, 2], [0, 1]))  # Pretty weird thing... To check
            theta = theta.swapaxes(1, 2).swapaxes(0, 1).swapaxes(2, 3)
            theta = theta.reshape(self.d * self._cutoff, self.d * self._cutoff)

        elif k == self._input_state.m - 2:
            theta = np.tensordot(self._sv_diag(k - 1), self._gamma[k, :], axes=(1, 0))
            theta = np.tensordot(theta, self._sv_diag(k), axes=(1, 0))
            theta = np.tensordot(theta, self._gamma[k + 1, :], axes=(2, 0))
            theta = np.tensordot(theta, self._transition_matrix(u), axes=([1, 3], [0, 1]))
            theta = theta.swapaxes(1, 2).swapaxes(0, 1).swapaxes(2, 3)
            theta = theta.reshape(self.d * self._cutoff, self.d * self._cutoff)

        v, s, w = np.linalg.svd(theta)

        v = v.reshape(self.d, self._cutoff, self.d * self._cutoff).swapaxes(0, 1).swapaxes(1, 2)[:, :self._cutoff]
        w = w.reshape(self.d * self._cutoff, self.d, self._cutoff).swapaxes(1, 2)[:self._cutoff]
        s = s[:self._cutoff]

        self.sv[k] = np.where(s > self._s_min, s, 0)

        if k > 0:
            rank = np.nonzero(self.sv[k - 1])[0][-1] + 1
            self._gamma[k, :rank] = v[:rank] / self.sv[k - 1, :rank][:, np.newaxis, np.newaxis]
            self._gamma[k, rank:] = 0
        else:
            self._gamma[k] = v
        if k < self._input_state.m - 2:
            rank = np.nonzero(self.sv[k + 1])[0][-1] + 1
            self._gamma[k + 1, :, :rank] = (w[:, :rank] / self.sv[k + 1, :rank][:, np.newaxis])
            self._gamma[k + 1, :, rank:] = 0
        else:
            self._gamma[k + 1] = w

    def _transition_matrix(self, u):
        "This function computes the elements (I,J) = (i_k, i_k+1, j_k, j_k+1) of the matrix U_k,k+1."
        d = self.d
        big_u = np.zeros((d,d,d,d), dtype = 'complex_')
        for i1 in range(d):
            for i2 in range(d):
                itot = i1 + i2
                u1, u2, u3, u4 = u[0,0], u[0,1], u[1, 0], u[1, 1]
                outputs = np.zeros((d,d), dtype = 'complex_')
                if itot <= self.n:
                    for k1 in range(i1+1):
                        for k2 in range(i2+1):
                            outputs[k1 + k2, itot - (k1 + k2)] += comb(i1, k1)*comb(i2, k2)\
                            *(u1**k1*u2**k2*u3**(i1-k1)*u4**(i2-k2))\
                            *(np.sqrt(factorial(k1+k2)*factorial(itot-k1-k2)))

                big_u[i1,i2,:] = outputs/(np.sqrt(factorial(i1)*factorial(i2)))
        return big_u

    def _sv_diag(self, k):
        if self._res[self._current_input]["sv"].any():
            sv = self._res[self._current_input]["sv"]
        else:
            sv = self.sv
        sv_diag = np.zeros((self._cutoff, self._cutoff))
        np.fill_diagonal(sv_diag, sv[k, :])
        return sv_diag
