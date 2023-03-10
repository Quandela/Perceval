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

import pytest
import perceval as pcvl
import sympy as sp


def test_format_simple():
    assert pcvl.simple_float(1)[1] == "1"
    assert pcvl.simple_float(-0)[1] == "0"
    assert pcvl.simple_float(0.0000000000001)[1] != "1e-10"
    assert pcvl.simple_float(1.0000000000001)[1] == "1"
    assert pcvl.simple_float(-2/3)[1] == "-2/3"
    assert pcvl.simple_float(-2/3, nsimplify=False)[1] == "-0.666667"
    assert pcvl.simple_float(-2/3, nsimplify=False, precision=1e-7)[1] == "-0.6666667"
    assert pcvl.simple_float(-2/30000, nsimplify=False, precision=1e-7)[1] == "-6.6666667e-5"
    assert pcvl.simple_float(float(-23*sp.pi/19))[1] == "-23*pi/19"


def test_format_complex():
    assert pcvl.simple_complex(1)[1] == "1"
    assert pcvl.simple_complex(-0)[1] == "0"
    assert pcvl.simple_complex(0.0000000000001)[1] != "1e-10"
    assert pcvl.simple_complex(1.0000000000001)[1] == "1"
    assert pcvl.simple_complex(-2j/3)[1] == "-2*I/3"
    assert pcvl.simple_complex(complex(1/sp.sqrt(2)-5j*sp.sqrt(5)/3))[1] == "sqrt(2)/2-5*sqrt(5)*I/3"
    assert pcvl.simple_complex(0.001+1e-15j)[1] == "0.001"
    assert pcvl.simple_complex(0.0001+1e-15j)[1] == "1e-4"


def test_format_pdisplay(capfd):
    pcvl.pdisplay(0.50000000001)
    out, err = capfd.readouterr()
    assert out.strip() == "1/2"
    pcvl.pdisplay(0.5001)
    out, err = capfd.readouterr()
    assert out.strip() == "0.5001"
    pcvl.pdisplay(0.5001, precision=1e-3)
    out, err = capfd.readouterr()
    assert out.strip() == "1/2"
    pcvl.pdisplay(1j)
    out, err = capfd.readouterr()
    assert out.strip() == "I"
