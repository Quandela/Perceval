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
    matched = bs.match(symb.BS(theta=pcvl.P("theta")))
    assert matched is not None
    assert pytest.approx(0.865743) == matched.v_map.get("theta", None)
    assert matched.pos_map == {0: 0}


def test_match_nomatch():
    bs = phys.PS(phi=0.3)
    assert bs.match(phys.BS(theta=pcvl.P("theta"))) is None


def test_match_perm():
    bs = phys.PERM([1, 0])
    pattern = phys.BS(theta=pcvl.P("theta"), phi_a=pcvl.P("phi_a"),
                                  phi_b=pcvl.P("phi_b"), phi_d=pcvl.P("phi_d"))
    matched = bs.match(pattern)
    assert matched is not None
    theta = matched.v_map.get("theta", None)
    assert pytest.approx(np.pi/2) == float(theta) or pytest.approx(3*np.pi/2) == float(theta)


def test_match_double():
    bs = phys.BS() // phys.PS(0.5)
    pattern = phys.BS() // phys.PS(pcvl.P("phi"))
    matched = bs.match(pattern)
    assert matched is not None
    print(matched)
    assert pytest.approx(1/2) == matched.v_map.get("phi", None)
    assert matched.pos_map == {0:0, 1:1}
    bs = phys.BS() // (1, phys.PS(0.5))
    pattern = phys.BS() // phys.PS(pcvl.P("phi"))
    matched = bs.match(pattern)
    assert not matched


def test_match_rec():
    mzi = phys.BS() // phys.PS(0.5) // phys.BS() // (1, phys.PS(0.3))
    pattern = phys.BS() // (1, phys.PS(pcvl.P("phi")))
    matched = mzi.match(pattern)
    assert matched is None
    pattern = phys.BS() // (1, phys.PS(pcvl.P("phi")))
    matched = mzi.match(pattern, browse=True)
    assert matched is not None
    assert matched.pos_map == {2:0, 3:1}


def test_match_rec_inv():
    c = pcvl.Circuit(3)//(1,phys.BS())//(0,phys.BS())//(1,phys.BS())
    pattern=pcvl.Circuit(3)//(0,phys.BS())//(1,phys.BS())//(0,phys.BS())
    matched = c.match(pattern)
    assert matched is None


def test_match_simple_seq():
    p2 = phys.BS() // phys.BS()
    c = phys.BS() // phys.BS()
    matched = c.match(p2)
    assert matched
    assert matched.pos_map == {0:0, 1:1}


def test_subnodes_0():
    bs = phys.Circuit(2).add(0, phys.BS())
    assert bs.find_subnodes(0) == [None, None]
    bs = phys.Circuit(3).add(0, phys.BS()).add(1, phys.PS(0.2)).add(1, phys.BS())
    assert bs.find_subnodes(0) == [None, (1,0)]
    assert bs.find_subnodes(1) == [(2,0)]
