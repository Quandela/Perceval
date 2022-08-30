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
import perceval.components.base_components as comp

import numpy as np


def test_determinant_base():
    c = comp.GenericBS()
    assert abs(c.U.det().simplify()) == 1


def test_determinant_generic():
    c = comp.GenericBS(theta=pcvl.P("θ"), phi_a=pcvl.P("phi_a"), phi_b=pcvl.P("phi_b"), phi_d=pcvl.P("phi_d"))
    assert abs(c.U.det().simplify()) == 1


def test_determinant_1():
    c = comp.GenericBS(theta=pcvl.P("θ"), phi_a=np.pi/2, phi_b=np.pi/2, phi_d=0)
    assert abs(c.U.det().simplify()) == 1


def test_determinant_2():
    c = comp.GenericBS(theta=pcvl.P("θ"), phi_a=np.pi/2, phi_b=np.pi/2, phi_d=np.pi/2)
    assert abs(c.U.det().simplify()) == 1


def test_determinant_3():
    c = comp.GenericBS(theta=pcvl.P("θ"), phi_a=0, phi_b=0, phi_d=0)
    assert abs(c.U.det().simplify()) == 1
