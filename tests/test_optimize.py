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

from datetime import datetime

import pytest

import perceval as pcvl
from perceval.algorithm.optimize import optimize
from perceval.algorithm.norm import fidelity, frobenius
import perceval.components.base_components as comp



def test_optimize_fidelity():
    c = (pcvl.Circuit(3, name="rewrite")
         // (0, comp.PS(pcvl.P("beta2")))
         // (1, comp.PS(pcvl.P("beta1")))
         // (1, comp.GenericBS(theta=pcvl.P("alpha1")))
         // (0, comp.GenericBS(theta=pcvl.P("alpha2")))
         // (1, comp.PS(pcvl.P("beta3")))
         // (1, comp.GenericBS(theta=pcvl.P("alpha3")))
         // (0, comp.PS(pcvl.P("beta4")))
         // (1, comp.PS(pcvl.P("beta5")))
         // (2, comp.PS(pcvl.P("beta6"))))
    v = pcvl.Matrix.random_unitary(3)
    res = optimize(c, v, fidelity)
    assert pytest.approx(1) == res.fun


def test_optimize_frobenius():
    c = (pcvl.Circuit(3, name="rewrite")
         // (0, comp.PS(pcvl.P("beta2")))
         // (1, comp.PS(pcvl.P("beta1")))
         // (1, comp.GenericBS(theta=pcvl.P("alpha1")))
         // (0, comp.GenericBS(theta=pcvl.P("alpha2")))
         // (1, comp.PS(pcvl.P("beta3")))
         // (1, comp.GenericBS(theta=pcvl.P("alpha3")))
         // (0, comp.PS(pcvl.P("beta4")))
         // (1, comp.PS(pcvl.P("beta5")))
         // (2, comp.PS(pcvl.P("beta6"))))
    v = pcvl.Matrix.random_unitary(3)
    res = optimize(c, v, frobenius, sign=-1)
    # test that the frobenius norm is almost 0 (pytest.approx will not work with almost 0)
    assert pytest.approx(0.5) == res.fun+0.5
