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
import perceval.components.unitary_components as comp

import numpy as np


def test_determinant_base():
    c = comp.BS()
    assert abs(c.U.det().simplify()) == 1


def test_determinant_generic():
    c = comp.BS(theta=pcvl.P("θ"), phi_tl=pcvl.P("phi_tl"), phi_bl=pcvl.P("phi_bl"), phi_tr=pcvl.P("phi_tr"))
    assert abs(c.U.det().simplify()) == 1


def test_determinant_1():
    c = comp.BS(theta=pcvl.P("θ"), phi_tl=np.pi/2, phi_bl=np.pi/2, phi_tr=0)
    assert abs(c.U.det().simplify()) == 1


def test_determinant_2():
    c = comp.BS(theta=pcvl.P("θ"), phi_tl=np.pi/2, phi_bl=np.pi/2, phi_tr=np.pi/2)
    assert abs(c.U.det().simplify()) == 1
