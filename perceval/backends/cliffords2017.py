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

from .template import Backend

import numpy as np
import quandelibc as qc
from perceval.utils import BasicState


def _square(x):
    return abs(x**2).real

def _get_scale(w):
    return max([max(abs(x.real), abs(x.imag)) for x in w])

class CliffordClifford2017Backend(Backend):
    name = "CliffordClifford2017"
    supports_symbolic = False
    supports_circuit_computing = False

    def prob_be(self, input_state, output_state, n=None, output_idx=None):
        raise NotImplementedError

    def sample(self, input_state):
        # prepare Us that is a m*n matrix
        m = self._m
        n = input_state.n
        fs = [0]*m
        Us = np.zeros((n, m), dtype=np.complex128)
        # build Us while transposing it
        rowidx = 0
        for ik in range(self._m):
            for i in range(input_state[ik]):
                Us[rowidx, :] = self._U[:, ik]
                rowidx += 1
        if n > 1:
            A = Us[np.random.permutation(n), :]
        else:
            A = Us
        w = _square(A[0, :])
        mode_seq = [np.random.choice(np.arange(0, m), p=w/sum(w), size=1)[0]]
        fs[mode_seq[0]] = 1
        for mode_limit in range(2, n+1):
            # permanents of sub-matrices using Laplace-type expansion (arXiv:1505.05486)
            sub_perm = np.array(qc.sub_permanents_cx(np.copy(np.reshape(A[0:mode_limit, mode_seq],
                                                                        (-1, mode_limit-1)))))
            sub_perm /= _get_scale(sub_perm)
            # generate next mode from there
            perm_vector = np.dot(sub_perm.transpose(), A[0:mode_limit])
            w = _square(perm_vector)
            next_mode = np.random.choice(np.arange(0, m), p=w/sum(w), size=1)[0]
            mode_seq.append(next_mode)
            fs[next_mode] += 1
        return BasicState(fs)
