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

from perceval import Parameter
import perceval.components.unitary_components as comp
from perceval.rendering.pdisplay import pdisplay_matrix

import sympy as sp
import numpy as np


def test_definition():
    p = Parameter("alpha", 0)
    assert isinstance(p.spv, sp.Number)
    assert p.spv == 0
    assert p.defined and float(p) == 0


def test_variable():
    p = Parameter("alpha")
    assert isinstance(p.spv, sp.Expr)
    assert not p.defined


def test_set_variable():
    p = Parameter("alpha")
    p.set_value(0.5)
    assert isinstance(p.spv, sp.Number)
    assert p.defined
    assert float(p) == 0.5


def test_fixed_0():
    p = Parameter("alpha", 2)
    assert p.defined
    try:
        p.set_value(1)
    except RuntimeError:
        pass
    else:
        raise Exception("Cannot set a fixed parameter")


def test_fixed_1():
    p = Parameter("alpha")
    assert not p.fixed
    assert not p.defined
    p.set_value(1)
    assert not p.fixed
    assert p.defined


def test_basic_conv():
    # initially we were trying to convert numeric values into remarkable sympy expression, we are stopping that
    # due to overhead in sympy
    p = Parameter("R", 1/3)
    assert p._value == 1/3


def test_invalid_values():
    with pytest.raises(ValueError):
        Parameter("R", -1, 0, 1, False)
    with pytest.raises(ValueError):
        p = Parameter("R", None, 0, 1, False)
        p.set_value(-1)
    p = Parameter("R", None, 0, 1)
    p.set_value(0)


def test_periodic_values():
    p = Parameter("theta", 0, 0, 2*sp.pi)
    assert float(p)==0
    p = Parameter("theta", 5*np.pi/2, 0, 2 * sp.pi)
    assert float(p) == float(np.pi/2)


def test_multiple_parameter_use():
    phi = Parameter("phi")
    c = comp.BS.H(phi_bl=phi) // comp.BS.H(phi_tl=phi)
    assert pdisplay_matrix(c.U.simp()) == '''⎡exp(I*phi)/2 + 1/2  (exp(I*phi) - 1)*exp(I*phi)/2⎤
⎣exp(I*phi)/2 - 1/2  (exp(I*phi) + 1)*exp(I*phi)/2⎦'''
