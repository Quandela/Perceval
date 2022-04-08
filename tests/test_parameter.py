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

from perceval import Parameter

import sympy as sp


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


def test_basicconv():
    p = Parameter("R", 1/3)
    assert p._value == sp.S(1)/3
