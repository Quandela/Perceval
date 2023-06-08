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

from perceval.utils import mlstr
import numpy as np


def test_basic0():
    s = mlstr(0.234)
    assert str(s) == "0.234"


def test_basic1():
    s = mlstr()
    assert str(s) == ""


def test_basic2():
    s = mlstr("string")
    assert str(s) == "string"


def test_iadd0():
    s = mlstr("string")
    s += "123"
    assert str(s) == "string123"


def test_iadd1():
    s = mlstr("M = ")
    s += "|0 1|\n|1 0|"
    assert str(s) == "M = |0 1|\n    |1 0|"


def test_radd():
    assert str(1+mlstr("a\nb")) == "1a\n b"


def test_iadd_inv():
    s = "M = "
    s + mlstr("|0 1|\n|1 0|")


def test_format():
    s = mlstr("%s = 1/%f * %s")
    assert str(s % ("M", np.sqrt(2), "|0 1|\n|1 0|")) == "M = 1/1.414214 * |0 1|\n                 |1 0|"


def test_join():
    s = mlstr(" ").join(["a", "a\nb", "c", "a\ng\nc"])
    assert str(s) == "a a c a\n  b   g\n      c"
