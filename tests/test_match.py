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
from perceval.components.unitary_components import BS, PS, PERM
from perceval.utils.algorithms.optimize import optimize
from perceval.utils.algorithms import norm
from perceval.utils import random_seed


def setup_function(_):
    # The random seed is fixed for this test suite, because the optimization problem randomly fails to be accurate
    # enough
    random_seed(0)


def teardown_function(_):
    # Revert to an arbitrary random seed after these tests to not interfere with any subsequent call
    random_seed()


def _bs_rx(r):
    return BS(BS.r_to_theta(r))

def _bs_h(r):
    return BS.H(BS.r_to_theta(r))

def test_match_elementary():
    bs = _bs_rx(0.42)
    matched = bs.match(BS(theta=pcvl.P("theta")))
    assert matched is not None
    assert pytest.approx(1.7314869) == matched.v_map.get("theta", None)
    assert matched.pos_map == {0: 0}


def test_match_nomatch():
    bs = PS(phi=0.3)
    assert bs.match(BS.H(theta=pcvl.P("theta"))) is None


def test_match_perm():
    bs = PERM([1, 0])
    pattern = BS.H(theta=pcvl.P("theta"), phi_tl=pcvl.P("phi_tl"),
                   phi_bl=pcvl.P("phi_bl"), phi_tr=pcvl.P("phi_tr"))
    matched = bs.match(pattern)
    assert matched is not None
    theta = matched.v_map.get("theta", None)
    assert pytest.approx(np.pi) == float(theta) or pytest.approx(3*np.pi) == float(theta)


def test_match_double():
    bs = BS.H() // PS(0.5)
    pattern = BS.H() // PS(pcvl.P("phi"))
    matched = bs.match(pattern)
    assert matched is not None
    assert pytest.approx(1/2) == matched.v_map.get("phi", None)
    assert matched.pos_map == {0: 0, 1: 1}
    bs = BS.H() // (1, PS(0.5))
    pattern = BS.H() // PS(pcvl.P("phi"))
    matched = bs.match(pattern)
    assert not matched


def test_match_rec():
    mzi = BS.H() // PS(0.5) // BS.H() // (1, PS(0.3))
    pattern = BS.H() // (1, PS(pcvl.P("phi")))
    matched = mzi.match(pattern)
    assert matched is None
    pattern = BS.H() // (1, PS(pcvl.P("phi")))
    matched = mzi.match(pattern, browse=True)
    assert matched is not None
    assert matched.pos_map == {2: 0, 3: 1}


def test_match_rec_inv():
    c = pcvl.Circuit(3) // (1, BS.H()) // (0, BS.H()) // (1, BS.H())
    pattern = pcvl.Circuit(3) // (0, BS.H()) // (1, BS.H()) // (0, BS.H())
    matched = c.match(pattern)
    assert matched is None


def test_match_simple_seq():
    p2 = BS.H() // BS.H()
    c = BS.H() // BS.H()
    matched = c.match(p2)
    assert matched
    assert matched.pos_map == {0: 0, 1: 1}


def test_subnodes_0():
    bs = pcvl.Circuit(2).add(0, BS.H())
    assert bs.find_subnodes(0) == [None, None]
    bs = pcvl.Circuit(3).add(0, BS.H()).add(1, PS(0.2)).add(1, BS.H())
    assert bs.find_subnodes(0) == [None, (1, 0)]
    assert bs.find_subnodes(1) == [(2, 0)]


# def test_replace_R_by_theta_1():
#     p0a = BS(theta=pcvl.P("theta"))
#     p0b = BS(theta=pcvl.P("theta"), phi_tl=np.pi)
#     random_theta = (random.random()-0.5) * np.pi
#     a = BS(theta=random_theta)
#     matched = a.match(p0a)
#     if matched is None:
#         matched = a.match(p0b)
#         assert matched
#     assert pytest.approx(np.cos(random_theta)**2) == matched.v_map.get("R", None)


def test_match_rewrite_phase():
    a = PS(0.4) // PS(1.4)
    pattern2 = pcvl.Circuit(1, name="pattern") // PS(pcvl.P("phi1")) // PS(pcvl.P("phi2"))
    rewrite2 = pcvl.Circuit(1, name="rewrite") // PS(pcvl.P("phi"))
    matched = a.match(pattern2)
    for k, v in matched.v_map.items():
        pattern2.param(k).set_value(v)
    v = pattern2.compute_unitary(False)
    res = optimize(rewrite2, v, norm.frobenius, sign=-1)
    assert pytest.approx(0+1) == res.fun+1
    assert pytest.approx(v[0, 0]) == rewrite2.compute_unitary(False)[0, 0]


def test_match_switch_phases():
    a = pcvl.Circuit(2) // PS(0.4) // _bs_h(0.45)
    pattern3 = (pcvl.Circuit(2, name="pattern3") //
                (0, PS(pcvl.P("phi1"))) //
                (1, PS(pcvl.P("phip"))) //
                (0, _bs_h(0.45)))
    matched = a.match(pattern3, browse=True)
    assert matched is None

    a = pcvl.Circuit(2) // (0, PS(0.4)) // (1, PS(0.3)) // _bs_h(0.45)
    matched = a.match(pattern3, browse=True)
    assert matched is not None
    assert pytest.approx(0.4) == matched.v_map["phi1"]
    assert pytest.approx(0.3) == matched.v_map["phip"]
    pattern3_check = (pcvl.Circuit(2, name="pattern3") //
                      (1, PS(pcvl.P("phip"))) //
                      (0, _bs_h(0.45)))

    a = pcvl.Circuit(2) // (1, PS(0.3)) // (0, PS(0.4)) // _bs_h(0.45)
    matched = a.match(pattern3_check)
    assert matched is not None and matched.pos_map == {0: 0, 2: 1}
    matched = a.match(pattern3, browse=True)
    assert matched is not None
    assert pytest.approx(0.4) == matched.v_map["phi1"]
    assert pytest.approx(0.3) == matched.v_map["phip"]
