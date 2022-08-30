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

import random

import numpy as np
import pytest

import perceval as pcvl
import perceval.components.base_components as comp
from perceval.algorithm.optimize import optimize
from perceval.algorithm import norm
from perceval.utils import random_seed


def setup_function(_):
    # The random seed is fixed for this test suite, because the optimization problem randomly fails to be accurate
    # enough
    random_seed(0)


def teardown_function(_):
    # Revert to an arbitrary random seed after these tests to not interfere with any subsequent call
    random_seed()


def test_match_elementary():
    bs = comp.SimpleBS(R=0.42)
    matched = bs.match(comp.SimpleBS(theta=pcvl.P("theta")))
    assert matched is not None
    assert pytest.approx(0.865743) == matched.v_map.get("theta", None)
    assert matched.pos_map == {0: 0}


def test_match_nomatch():
    bs = comp.PS(phi=0.3)
    assert bs.match(comp.GenericBS(theta=pcvl.P("theta"))) is None


def test_match_perm():
    bs = comp.PERM([1, 0])
    pattern = comp.GenericBS(theta=pcvl.P("theta"), phi_a=pcvl.P("phi_a"),
                             phi_b=pcvl.P("phi_b"), phi_d=pcvl.P("phi_d"))
    matched = bs.match(pattern)
    assert matched is not None
    theta = matched.v_map.get("theta", None)
    assert pytest.approx(np.pi/2) == float(theta) or pytest.approx(3*np.pi/2) == float(theta)


def test_match_double():
    bs = comp.GenericBS() // comp.PS(0.5)
    pattern = comp.GenericBS() // comp.PS(pcvl.P("phi"))
    matched = bs.match(pattern)
    assert matched is not None
    assert pytest.approx(1/2) == matched.v_map.get("phi", None)
    assert matched.pos_map == {0: 0, 1: 1}
    bs = comp.GenericBS() // (1, comp.PS(0.5))
    pattern = comp.GenericBS() // comp.PS(pcvl.P("phi"))
    matched = bs.match(pattern)
    assert not matched


def test_match_rec():
    mzi = comp.GenericBS() // comp.PS(0.5) // comp.GenericBS() // (1, comp.PS(0.3))
    pattern = comp.GenericBS() // (1, comp.PS(pcvl.P("phi")))
    matched = mzi.match(pattern)
    assert matched is None
    pattern = comp.GenericBS() // (1, comp.PS(pcvl.P("phi")))
    matched = mzi.match(pattern, browse=True)
    assert matched is not None
    assert matched.pos_map == {2: 0, 3: 1}


def test_match_rec_inv():
    c = pcvl.Circuit(3) // (1, comp.GenericBS()) // (0, comp.GenericBS()) // (1, comp.GenericBS())
    pattern = pcvl.Circuit(3) // (0, comp.GenericBS()) // (1, comp.GenericBS()) // (0, comp.GenericBS())
    matched = c.match(pattern)
    assert matched is None


def test_match_simple_seq():
    p2 = comp.GenericBS() // comp.GenericBS()
    c = comp.GenericBS() // comp.GenericBS()
    matched = c.match(p2)
    assert matched
    assert matched.pos_map == {0: 0, 1: 1}


def test_subnodes_0():
    bs = pcvl.Circuit(2).add(0, comp.GenericBS())
    assert bs.find_subnodes(0) == [None, None]
    bs = pcvl.Circuit(3).add(0, comp.GenericBS()).add(1, comp.PS(0.2)).add(1, comp.GenericBS())
    assert bs.find_subnodes(0) == [None, (1, 0)]
    assert bs.find_subnodes(1) == [(2, 0)]


def test_replace_R_by_theta_1():
    p0a = comp.SimpleBS(R=pcvl.P("R"))
    p0b = comp.SimpleBS(R=pcvl.P("R"), phi=np.pi)
    random_theta = (random.random()-0.5) * np.pi
    a = comp.SimpleBS(theta=random_theta)
    matched = a.match(p0a)
    if matched is None:
        matched = a.match(p0b)
        assert matched
    assert pytest.approx(np.cos(random_theta)**2) == matched.v_map.get("R", None)


def test_match_rewrite_phase():
    a = comp.PS(0.4) // comp.PS(1.4)
    pattern2 = pcvl.Circuit(1, name="pattern") // comp.PS(pcvl.P("phi1")) // comp.PS(pcvl.P("phi2"))
    rewrite2 = pcvl.Circuit(1, name="rewrite") // comp.PS(pcvl.P("phi"))
    matched = a.match(pattern2)
    for k, v in matched.v_map.items():
        pattern2[k].set_value(v)
    v = pattern2.compute_unitary(False)
    res = optimize(rewrite2, v, norm.frobenius, sign=-1)
    assert pytest.approx(0+1) == res.fun+1
    assert pytest.approx(v[0, 0]) == rewrite2.compute_unitary(False)[0, 0]


def test_match_switch_phases():
    a = pcvl.Circuit(2) // comp.PS(0.4) // comp.GenericBS(R=0.45)
    pattern3 = (pcvl.Circuit(2, name="pattern3") //
                (0, comp.PS(pcvl.P("phi1"))) //
                (1, comp.PS(pcvl.P("phip"))) //
                (0, comp.GenericBS(R=0.45)))
    matched = a.match(pattern3, browse=True)
    assert matched is None

    a = pcvl.Circuit(2) // (0, comp.PS(0.4)) // (1, comp.PS(0.3)) // comp.GenericBS(R=0.45)
    matched = a.match(pattern3, browse=True)
    assert matched is not None
    assert pytest.approx(0.4) == matched.v_map["phi1"]
    assert pytest.approx(0.3) == matched.v_map["phip"]
    pattern3_check = (pcvl.Circuit(2, name="pattern3") //
                      (1, comp.PS(pcvl.P("phip"))) //
                      (0, comp.GenericBS(R=0.45)))

    a = pcvl.Circuit(2) // (1, comp.PS(0.3)) // (0, comp.PS(0.4)) // comp.GenericBS(R=0.45)
    matched = a.match(pattern3_check)
    assert matched is not None and matched.pos_map == {0: 0, 2: 1}
    matched = a.match(pattern3, browse=True)
    assert matched is not None
    assert pytest.approx(0.4) == matched.v_map["phi1"]
    assert pytest.approx(0.3) == matched.v_map["phip"]
