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
import sympy as sp

from perceval import Parameter, Expression
import perceval.components.unitary_components as comp
from perceval.rendering.pdisplay import pdisplay_matrix



def test_definition():
    p = Parameter("alpha", 0)
    assert isinstance(p.spv, sp.Number)
    assert p.spv == 0
    assert p.defined and float(p) == 0


def test_variable():
    p = Parameter("alpha")
    assert isinstance(p.spv, sp.Expr)
    assert not p.defined
    assert p.is_variable


def test_set_variable():
    p = Parameter("alpha")
    p.set_value(0.5)
    assert isinstance(p.spv, sp.Number)
    assert p.defined
    assert float(p) == 0.5


def test_fixed_0():
    p = Parameter("alpha", 2)
    assert p.fixed
    assert p.defined
    assert not p.is_variable
    with pytest.raises(RuntimeError):
        p.set_value(1)  # Cannot set value to a fixed parameter


def test_fixed_1():
    p = Parameter("alpha")
    assert not p.fixed
    assert not p.defined
    assert p.is_variable
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
    p = Parameter("theta", 5*math.pi/2, 0, 2 * sp.pi)
    assert float(p) == float(math.pi/2)


def test_multiple_parameter_use():
    phi = Parameter("phi")
    c = comp.BS.H(phi_bl=phi) // comp.BS.H(phi_tl=phi)
    assert pdisplay_matrix(c.U.simp()) == '''⎡exp(I*phi)/2 + 1/2  (exp(I*phi) - 1)*exp(I*phi)/2⎤
⎣exp(I*phi)/2 - 1/2  (exp(I*phi) + 1)*exp(I*phi)/2⎦'''


def test_expression_arithmetic():
    p_a = Parameter("A")
    p_b = Parameter("B")
    sum_ab = Expression("A + B", {p_a, p_b})

    p_c = Parameter("C")
    sum_abc = sum_ab + p_c

    assert "A + B + C" in sum_abc.name
    a = 5
    b = 6
    c = 1
    p_a.set_value(a)
    p_b.set_value(b)
    p_c.set_value(c)
    assert float(sum_abc) == a + b + c

    diff_ab = Expression("A - B", {p_a, p_b})
    diff_ab_over_c = diff_ab / p_c
    assert float(diff_ab_over_c) == (a - b) / c

    p_d = Parameter("D")
    diff_cd = Expression("C - D", {p_c, p_d})
    d = 8
    diff_ab_over_diff_cd = diff_ab / diff_cd
    p_d.set_value(d)
    new_a = 2
    p_a.set_value(new_a)

    assert float(diff_ab_over_diff_cd) == (new_a - b) / (c - d)


def test_expression_missing_parameter():
    p_a = Parameter("A")
    with pytest.raises(RuntimeError):
        Expression("A + B", {p_a})


def test_expression_math_functions():
    p_a = Parameter("A")
    cos_a = Expression("cos(A)", {p_a})
    a = math.pi / 3
    p_a.set_value(a)
    assert float(cos_a) == math.cos(a)

    cos_a_sq = cos_a**2
    assert float(cos_a_sq) == math.cos(a) ** 2

def test_expression_parameter_retrieval():
    param_names = ["A", "B", "CCC", "Day", "theta"]
    params = set()
    for n in param_names:
        params.add(Parameter(n))
    sum_params = Expression("+".join(param_names), params)  # Builds the sum of all params
    param_list = sum_params.parameters
    assert len(param_list) == len(param_names)
    for p in param_list:
        assert p.name in param_names
