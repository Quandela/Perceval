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
from _test_utils import assert_sv_close
from perceval.utils import StateVector
from perceval.simulators._simulator_utils import _merge_sv
from math import sqrt


def test_merge_sv():
    a1 = complex(0, 1/sqrt(2))
    b1 = complex(-1.2, 3.7)
    sv1 = a1 * StateVector([1, 0]) + b1 * StateVector([0, 1])

    a2 = complex(-0.598, -0.65)
    b2 = complex(0.15, 0.297)
    sv2 = a2 * StateVector([1, 0]) + b2 * StateVector([0, 1])

    sv_res = _merge_sv(sv1, sv2)
    assert_sv_close(
        sv_res,
        a1*a2 * StateVector([2, 0]) + (a1*b2 + b1*a2) * StateVector([1, 1]) + b1*b2 * StateVector([0, 2])
    )
