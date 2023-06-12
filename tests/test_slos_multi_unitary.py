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

import perceval as pcvl
import pytest
import numpy as np

def test_multi_unitary_0():
    N = 2
    m = 3
    u1 = pcvl.Matrix.random_unitary(m)
    u2 = pcvl.Matrix.random_unitary(m)
    input_state = pcvl.BasicState([1] * N + [0] * (m - N))
    simulator_backend = pcvl.BackendFactory().get_backend("SLOS")
    s1 = simulator_backend(u1)
    p1 = s1.all_prob(input_state)

    s2 = simulator_backend(u2)
    p2 = s2.all_prob(input_state)

    s1.set_multi_unitary([u1, u2])
    p1a, p2a = s1.all_prob(input_state)

    assert pytest.approx(np.linalg.norm(p1-p1a)) == 0, "p1 and p1a should be equal - %f" % np.linalg.norm(p1-p1a)
    assert pytest.approx(np.linalg.norm(p2-p2a)) == 0, "p2 and p2a should be equal - %f" % np.linalg.norm(p2-p2a)

    s1.U = u1
    p1b = s1.all_prob(input_state)
    assert pytest.approx(np.linalg.norm(p1-p1b)) == 0, "p1 and p1b should be equal - %f" % np.linalg.norm(p1-p1b)
