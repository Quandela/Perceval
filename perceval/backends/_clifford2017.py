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
import exqalibur as xq
from perceval.utils import BasicState

from ._abstract_backends import ASamplingBackend


def _square(x):
    return abs(x**2).real


def _get_scale(w):
    return max([max(abs(x.real), abs(x.imag)) for x in w])


class Clifford2017Backend(ASamplingBackend):

    @property
    def name(self) -> str:
        return "CliffordClifford2017"

    def _prepare_us(self):
        # prepare Us that is a m*n matrix
        m = self._input_state.m
        us = np.zeros((self._input_state.n, m), dtype=np.complex128)
        # build Us while transposing it
        rowidx = 0
        for ik in range(m):
            extract = self._umat[:, ik]
            for _ in range(self._input_state[ik]):
                us[rowidx, :] = extract
                rowidx += 1
        return us

    def sample(self) -> BasicState:
        n = self._input_state.n
        if n == 0:
            return self._input_state

        A = self._prepare_us()
        if n > 1:
            A = A[np.random.permutation(n), :]
        w = _square(A[0, :])
        m = self._input_state.m
        mode_seq = [np.random.choice(np.arange(0, m), p=w / sum(w), size=1)[0]]
        output_state = [0] * m
        output_state[mode_seq[0]] = 1
        for mode_limit in range(2, n + 1):
            # permanents of sub-matrices using Laplace-type expansion (arXiv:1505.05486)
            sub_perm = np.array(
                xq.sub_permanents_cx(np.copy(np.reshape(A[0:mode_limit, mode_seq], (-1, mode_limit - 1)))))
            sub_perm /= _get_scale(sub_perm)
            # generate next mode from there
            perm_vector = np.dot(sub_perm.transpose(), A[0:mode_limit])
            w = _square(perm_vector)
            next_mode = np.random.choice(np.arange(0, m), p=w/sum(w), size=1)[0]
            mode_seq.append(next_mode)
            output_state[next_mode] += 1
        return BasicState(output_state)
