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

from perceval.backends import ASamplingBackend
from perceval.utils import BasicState, BSCount, SVDistribution


class SamplesProvider:

    def __init__(self, sampling_backend: ASamplingBackend):
        self._backend = sampling_backend
        self._pools = {}
        self._weights = BSCount()
        self._sample_coeff = 1.1
        self._min_samples = 100

    def prepare(self, noisy_input: SVDistribution, n_samples: int):
        for sv, prob in noisy_input.items():
            ns = int(max(prob * self._sample_coeff * n_samples, self._min_samples))
            for bs in sv[0].separate_state(keep_annotations=False):
                self._weights.add(bs, ns)

        for input_state, count in self._weights.items():
            if input_state.n == 0:
                self._pools[input_state] = [input_state]*count
            else:
                self._backend.set_input_state(input_state)
                self._pools[input_state] = self._backend.samples(count)
            self._weights[input_state] = int(0.1 * self._weights[input_state])

    def _compute_samples(self, fock_state: BasicState):
        if fock_state not in self._pools:
            self._pools[fock_state] = []
            self._weights[fock_state] = self._min_samples

        self._backend.set_input_state(fock_state)
        self._pools[fock_state] += self._backend.samples(self._weights[fock_state])
        self._weights[fock_state] = int(self._weights[fock_state] * self._sample_coeff)

    def sample_from(self, input_state: BasicState) -> BasicState:
        if input_state not in self._pools or len(self._pools[input_state]) == 0:
            self._compute_samples(input_state)
        return self._pools[input_state].pop()
