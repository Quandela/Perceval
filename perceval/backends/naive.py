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

import math
import numpy as np

from .template import Backend
import quandelibc as qc


class NaiveBackend(Backend):
    """Naive algorithm, no clever calculation path, does not cache anything,
       recompute all states on the fly"""
    name = "Naive"
    supports_symbolic = False
    supports_circuit_computing = False

    def probampli_be(self, input_state, output_state, n=None, output_idx=None):
        if input_state.n != output_state.n:
            return 0
        if n is None:
            n = input_state.n
        Ust = np.empty((n, n), dtype=complex)
        colidx = 0
        p = 1
        for ok in range(self._realm):
            p *= math.factorial(output_state[ok])
        for ik in range(self._realm):
            p *= math.factorial(input_state[ik])
            for i in range(input_state[ik]):
                rowidx = 0
                for ok in range(self._realm):
                    for j in range(output_state[ok]):
                        Ust[rowidx, colidx] = self._U[ok, ik]
                        rowidx += 1
                colidx += 1
        return qc.permanent_cx(Ust, n_threads=1)/math.sqrt(p)

    def prob_be(self, input_state, output_state, n=None, output_idx=None):
        return abs(self.probampli_be(input_state, output_state, n, output_idx))**2
