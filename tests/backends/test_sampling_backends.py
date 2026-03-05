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
import pytest

from perceval.backends import ASamplingBackend, BackendFactory
from perceval.components import catalog
from perceval.utils import BasicState

@pytest.mark.long_test
@pytest.mark.parametrize("backend_name", ["CliffordClifford2017", "Stepper"])
def test_backend_cnot(backend_name):
    # Two last modes are ancillaries
    s00 = BasicState([1, 0, 1, 0, 0, 0])
    s01 = BasicState([1, 0, 0, 1, 0, 0])
    s10 = BasicState([0, 1, 1, 0, 0, 0])
    s11 = BasicState([0, 1, 0, 1, 0, 0])

    expected = [
        [ s00, s00 ],
        [ s01, s01 ],
        [ s10, s11 ],
        [ s11, s10 ],
    ]
    backend: ASamplingBackend = BackendFactory.get_backend(backend_name)
    cnot = catalog["postprocessed cnot"].build_circuit()
    backend.set_circuit(cnot)

    N = 1000
    for input, output in expected:
        backend.set_input_state(input)
        unknown = set()
        correct = 0
        for _ in range(N):
            bs = backend.sample()
            if bs == output:
                correct += 1
            elif bs[4] or bs[5] or bs[0] + bs[1] != 0 or bs[2] + bs[3] != 0:
                pass # post-processed
            else:
                unknown.add(bs)
        assert len(unknown) == 0
        assert correct/N == pytest.approx(1/9, abs = 2.5758 * math.sqrt(8/81 / N)), "correct sample proportion out of 99% confidence interval"
