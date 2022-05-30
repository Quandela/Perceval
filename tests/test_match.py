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

import perceval as pcvl
import perceval.lib.phys as phys
import perceval.lib.symb as symb
import numpy as np

def test_match_elementary():
    bs = symb.BS(R=0.42)
    matched, c = bs.match(symb.BS(theta=pcvl.P("theta")))
    assert matched and pytest.approx(0.865743) == float(c["theta"])


def test_match_nomatch():
    bs = phys.PS(phi=0.3)
    matched, c = bs.match(phys.BS(theta=pcvl.P("theta")))
    assert not matched


def test_match_perm():
    bs = phys.PERM([1, 0])
    matched, c = bs.match(phys.BS(theta=pcvl.P("theta"), phi_a=pcvl.P("phi_a"),
                                  phi_b=pcvl.P("phi_b"), phi_d=pcvl.P("phi_d")))
    assert matched
    assert pytest.approx(np.pi/2) == float(c["theta"]) or pytest.approx(3*np.pi/2) == float(c["theta"])
