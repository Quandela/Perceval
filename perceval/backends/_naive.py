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

import math
import numpy as np

import exqalibur as xq
from ._abstract_backends import AProbAmpliBackend
from perceval.utils import BasicState


class NaiveBackend(AProbAmpliBackend):
    """Naive algorithm, no clever calculation path, does not cache anything,
       recompute all states on the fly"""

    @property
    def name(self) -> str:
        return "Naive"

    def prob_amplitude(self, output_state: BasicState) -> complex:
        n = self._input_state.n
        m = self._input_state.m
        if n != output_state.n:
            return complex(0)
        if n == 0:
            return complex(1)
        u_st = np.empty((n, n), dtype=complex)
        colidx = 0
        p = output_state.prodnfact() * self._input_state.prodnfact()
        for ik in range(m):
            for i in range(self._input_state[ik]):
                rowidx = 0
                for ok in range(m):
                    for j in range(output_state[ok]):
                        u_st[rowidx, colidx] = self._umat[ok, ik]
                        rowidx += 1
                colidx += 1
        return xq.permanent_cx(u_st, n_threads=1)/math.sqrt(p)
