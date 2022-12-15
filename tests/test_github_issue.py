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

import pytest

import numpy as np

import perceval as pcvl
import perceval.components.unitary_components as comp


def test_34():
    bs = comp.BS.H(theta=0)
    c = pcvl.Circuit(4, name='phase')
    c.add((2, 3), bs)
    pcvl.pdisplay(c)  # looks good
    simulator_backend = pcvl.BackendFactory.get_backend("Naive")
    simu = simulator_backend(c.compute_unitary())
    state = pcvl.BasicState([1, 1, 1, 1])
    pa = simu.probampli(state, pcvl.BasicState([1, 1, 1, 1]))
    assert pytest.approx(-1) == pa.real
    assert pytest.approx(1) == abs(pa)
