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

import perceval as pcvl
from perceval.utils.statevector import convert_polarized_state, build_spatial_output_states
from perceval.rendering.pdisplay import pdisplay_matrix
import pytest

import sympy as sp
import perceval.components.unitary_components as comp


def test_polar_parse_error():
    invalid_str = {"a": "angle value should not contain variable",
                   "(1,2,3)": "more than two parameters",
                   "(1,2": "missing closing parenthesis",
                   "2*pi": "theta should be in [0,pi]"}
    for s, error in invalid_str.items():
        with pytest.raises(ValueError) as parse_error:
            pcvl.Polarization.parse(s)
        assert str(parse_error.value).find(error) != -1, "'%s' - does not match '%s'" % (str(parse_error), error)


def test_polar_parse_ok():
    valid_str = {"H": (0, 0, "H"),
                 "V": (sp.pi, 0, "V"),
                 "D": (sp.pi/2, 0, "D"),
                 "A": (sp.pi/2, sp.pi, "A"),
                 "R": (sp.pi/2, 3*sp.pi/2, "R"),
                 "L": (sp.pi/2, sp.pi/2, "L"),
                 "0": (0, 0, "H"),
                 "(0,0)": (0, 0, "H"),
                 "pi": (sp.pi, 0, "V"),
                 "(pi,0)": (sp.pi, 0, "V"),
                 "(pi/2,pi)": (sp.pi/2, sp.pi, "A"),
                 "(pi/2,3*pi/2)": (sp.pi / 2, 3*sp.pi/2, "R"),
                 "(pi/2,1/2*pi)": (sp.pi / 2, sp.pi/2, "L"),
                 "(pi/4,0)": (sp.pi/4, 0, "pi/4"),
                 "(pi/4,pi/2)": (sp.pi / 4, sp.pi/2, "(pi/4,pi/2)")}
    for k, (theta, phi, s) in valid_str.items():
        p = pcvl.Polarization.parse(k)
        assert pytest.approx(float(theta)) == float(p.theta_phi[0])
        assert pytest.approx(float(phi)) == float(p.theta_phi[1])
        assert str(p) == s


def test_polar_init():
    p = pcvl.Polarization("H")
    assert p.theta_phi[0] == 0
    assert p.theta_phi[1] == 0
    assert str(p) == "H"


def test_polar_circuit1():
    c = pcvl.Circuit(2)
    c.add(0, comp.PS(phi=sp.pi/2))
    assert not c.requires_polarization
    c = pcvl.Circuit(2)
    c.add(0, comp.WP(sp.pi/3, sp.pi/2))
    assert c.requires_polarization


def test_polar_nmode():
    c = comp.BS.H()
    u = c.compute_unitary()
    pu = c.compute_unitary(use_polarization=True)
    assert u.shape == (2, 2)
    assert pu.shape == (4, 4)
    assert (pu[0::2, 0::2] == u).all()
    assert (pu[1::2, 1::2] == u).all()


def test_polar_circuit2():
    c = pcvl.Circuit(2)
    c //= comp.BS.H()
    c //= (1, comp.WP(sp.pi/4, sp.pi/2))
    u = c.compute_unitary(use_symbolic=True, use_polarization=True)
    assert u.shape == (4, 4)
    assert u[0, 0] == sp.sqrt(2)/2
    assert (u[2, 0] - 1/2+sp.I/2).simplify() == 0


def test_prep_state():
    s, m = convert_polarized_state(pcvl.BasicState("|{P:H},{P:V},0,{P:A}>"))
    assert str(s) == "|1,0,1,0,0,0,1,0>"
    assert pdisplay_matrix(m) == """
            ⎡1  0  0  0   0  0  0           0        ⎤
            ⎢0  1  0  0   0  0  0           0        ⎥
            ⎢0  0  0  -1  0  0  0           0        ⎥
            ⎢0  0  1  0   0  0  0           0        ⎥
            ⎢0  0  0  0   1  0  0           0        ⎥
            ⎢0  0  0  0   0  1  0           0        ⎥
            ⎢0  0  0  0   0  0  sqrt(2)/2   sqrt(2)/2⎥
            ⎣0  0  0  0   0  0  -sqrt(2)/2  sqrt(2)/2⎦
    """.strip().replace("            ", "")
    s2, m2 = convert_polarized_state(pcvl.BasicState("|{P:H}{P:H},{P:V},0,{P:A}>"))
    assert str(s2) == "|2,0,1,0,0,0,1,0>"
    assert (m2-m).all() == 0


def test_prep_multi_state():
    convert_polarized_state(pcvl.BasicState("|{P:H}{P:V},{P:V},0,{P:A}>"))


def test_convert_multistate():
    input_state, prep_matrix = convert_polarized_state(pcvl.BasicState("|2{P:H}3{P:V}>"))
    assert str(input_state) == "|2,3>"


def test_convert_multistate_nonorthogonal():
    with pytest.raises(ValueError):
        convert_polarized_state(pcvl.BasicState("|2{P:H}3{P:D}>"))


def test_build_spatial_output():
    assert sorted([str(s) for s in build_spatial_output_states(pcvl.BasicState("|2,0,1>"))]) == [
         '|0,2,0,0,0,1>',
         '|0,2,0,0,1,0>',
         '|1,1,0,0,0,1>',
         '|1,1,0,0,1,0>',
         '|2,0,0,0,0,1>',
         '|2,0,0,0,1,0>'
    ]


def test_subcircuit_polarization():
    a = pcvl.Circuit(2) // comp.PBS() // comp.PBS()
    assert a.requires_polarization, "subcircuit does not propagate polarization state"
    b = comp.BS.H() // a // a // comp.BS.H()
    assert b.requires_polarization, "subcircuit does not propagate polarization state"
