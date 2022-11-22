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

from typing import Union
import copy

import numpy as np
from math import factorial
from scipy.special import comb
from collections import defaultdict

from .template import Backend
from perceval.utils import BasicState, Matrix
from perceval.components import ACircuit


class MPSBackend(Backend):
    """Step-by-step circuit propagation algorithm, works on a circuit. Approximate the probability amplitudes with a cutoff.
    - For now only supports Phase shifters and Beam Splitters
    - TODO: link to the quandelibc computation
    """

    def __init__(self,
                 cu: Union[ACircuit, Matrix],
                 use_symbolic: bool = None,
                 n: int = None,
                 mask: list = None):
        super().__init__(cu, use_symbolic, n, mask)
        for r, c in self._C:
            assert c.compute_unitary(use_symbolic=False).shape[0] <= 2,\
                "MPS backend can not be used with components of using more than 2 modes"
        self._s_min = 1e-8
        self.cutoff = self.m
        self.res = defaultdict(lambda: defaultdict(lambda: np.array([0])))
        self.current_input = None

    name = "MPS"
    supports_symbolic = False
    supports_circuit_computing = True

    def set_cutoff(self, cutoff: int):
        assert isinstance(cutoff, int), "cutoff must be an integer"
        self.cutoff = cutoff

    def apply(self, r, c):
        u = c.compute_unitary(False)
        k_mode = r[0]
        if len(u) == 2:
            self.update_state(k_mode, u)  # --> quandelibc
        elif len(u) == 1:
            self.update_state_1_mode(k_mode, u)  # --> quandelibc

    def compile(self, input_state: BasicState) -> bool:
        var = [float(p) for p in self._C.get_parameters()]
        if self._compiled_input and self._compiled_input[0] == var and input_state in self.res:
            return False
        self._compiled_input = copy.copy((var, input_state))
        self.current_input = None

        # TODO : allow any StateVector as in stepper, or a list as in SLOS
        input_state *= BasicState([0] * (self.m - input_state.m))
        self.n = input_state.n
        self.d = self.n + 1
        self.cutoff = min(self.cutoff, self.d ** (self.m//2))
        self.gamma = np.zeros((self.m, self.cutoff, self.cutoff, self.d), dtype='complex_')
        for i in range(self.m):
            self.gamma[i, 0, 0, input_state[i]] = 1
        self.sv = np.zeros((self.m, self.cutoff))
        self.sv[:, 0] = 1

        for r, c in self._C:
            self.apply(r, c)

        self.res[tuple(input_state)]["gamma"] = self.gamma.copy()
        self.res[tuple(input_state)]["sv"] = self.sv.copy()
        return True

    def prob_be(self, input_state, output_state, n=None, output_idx=None):
        return abs(self.probampli_be(input_state, output_state, n, output_idx))**2

    def probampli_be(self, input_state, output_state, n=None, output_idx=None):
        # TODO: put in quandelibc
        mps_in_list = []
        self.current_input = tuple(input_state)
        for k in range(self.m - 1):
            mps_in_list.append(self.res[tuple(input_state)]["gamma"][k, :, :, output_state[k]])
            mps_in_list.append(self._sv_diag(k))
        mps_in_list.append(self.res[tuple(input_state)]["gamma"][self.m-1, :, :, output_state[self.m-1]])
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
        self.gamma[k] = np.tensordot(self.gamma[k], self._transition_matrix_1_mode(u), axes=(2,0))

    def update_state(self, k, u):

        if 0 < k < self.m - 2:
            theta = np.tensordot(self._sv_diag(k - 1), self.gamma[k, :], axes=(1, 0))
            theta = np.tensordot(theta, self._sv_diag(k), axes=(1, 0))
            theta = np.tensordot(theta, self.gamma[k + 1, :], axes=(2, 0))
            theta = np.tensordot(theta, self._sv_diag(k + 1), axes=(2, 0))
            theta = np.tensordot(theta, self._transition_matrix(u), axes=([1, 2], [0, 1]))
            theta = theta.swapaxes(1, 2).swapaxes(0, 1).swapaxes(2, 3)
            theta = theta.reshape(self.d * self.cutoff, self.d * self.cutoff)

        elif k == 0:
            theta = np.tensordot(self.gamma[k, :], self._sv_diag(k), axes=(1, 0))
            theta = np.tensordot(theta, self.gamma[k + 1, :], axes=(2, 0))
            theta = np.tensordot(theta, self._sv_diag(k + 1), axes=(2, 0))
            theta = np.tensordot(theta, self._transition_matrix(u),
                                 axes=([1, 2], [0, 1]))  # Pretty weird thing... To check
            theta = theta.swapaxes(1, 2).swapaxes(0, 1).swapaxes(2, 3)
            theta = theta.reshape(self.d * self.cutoff, self.d * self.cutoff)

        elif k == self.m - 2:
            theta = np.tensordot(self._sv_diag(k - 1), self.gamma[k, :], axes=(1, 0))
            theta = np.tensordot(theta, self._sv_diag(k), axes=(1, 0))
            theta = np.tensordot(theta, self.gamma[k + 1, :], axes=(2, 0))
            theta = np.tensordot(theta, self._transition_matrix(u), axes=([1, 3], [0, 1]))
            theta = theta.swapaxes(1, 2).swapaxes(0, 1).swapaxes(2, 3)
            theta = theta.reshape(self.d * self.cutoff, self.d * self.cutoff)

        v, s, w = np.linalg.svd(theta)

        v = v.reshape(self.d, self.cutoff, self.d * self.cutoff).swapaxes(0, 1).swapaxes(1, 2)[:, :self.cutoff]
        w = w.reshape(self.d * self.cutoff, self.d, self.cutoff).swapaxes(1, 2)[:self.cutoff]
        s = s[:self.cutoff]

        self.sv[k] = np.where(s > self._s_min, s, 0)

        if k > 0:
            rank = np.nonzero(self.sv[k - 1])[0][-1] + 1
            self.gamma[k, :rank] = v[:rank] / self.sv[k - 1, :rank][:, np.newaxis, np.newaxis]
            self.gamma[k, rank:] = 0
        else:
            self.gamma[k] = v
        if k < self.m - 2:
            rank = np.nonzero(self.sv[k + 1])[0][-1] + 1
            self.gamma[k + 1, :, :rank] = (w[:, :rank] / self.sv[k + 1, :rank][:, np.newaxis])
            self.gamma[k + 1, :, rank:] = 0
        else:
            self.gamma[k + 1] = w

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
        if self.res[self.current_input]["sv"].any():
            sv = self.res[self.current_input]["sv"]
        else:
            sv = self.sv
        sv_diag = np.zeros((self.cutoff, self.cutoff))
        np.fill_diagonal(sv_diag, sv[k, :])
        return sv_diag
