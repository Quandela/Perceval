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
from typing import List, Tuple

import exqalibur as xq
from . import NaiveBackend
from perceval.utils import BasicState

class NaiveApproxBackend(NaiveBackend):
    """Naive algorithm with Gurvits computations of permanents"""

    def __init__(self, gurvits_iterations = 10000):
        self._gurvits_iterations = gurvits_iterations
        NaiveBackend.__init__(self)

    @property
    def name(self) -> str:
        return "NaiveApprox"

    def _compute_permanent(self, M):
        permanent_with_error = xq.estimate_permanent_cx(M, self._gurvits_iterations, 0)
        return permanent_with_error[0]

    def prob_amplitude_with_error(self, output_state: BasicState) -> Tuple[complex, float]:
        M = self._compute_submatrix(output_state)
        permanent_with_error = xq.estimate_permanent_cx(M, self._gurvits_iterations, 0)
        normalization_coeff = math.sqrt(output_state.prodnfact() * self._input_state.prodnfact())
        return (permanent_with_error[0]/normalization_coeff, permanent_with_error[1]/normalization_coeff) \
            if M.size > 1 else (M[0, 0], 0)

    def probability_confidence_interval(self, output_state: BasicState) -> List[float]:
        mean, err = self.prob_amplitude_with_error(output_state)
        min_prob = max((abs(mean) - err) ** 2, 0)
        max_prob = min((abs(mean) + err) ** 2, 1)
        return [min_prob, max_prob]
