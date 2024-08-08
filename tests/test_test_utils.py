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

from perceval.utils import StateVector

from _test_utils import assert_sv_close


def test_utils():
    sv_0_1 = StateVector([0, 1])
    sv_1_0 = StateVector([1, 0])
    sv_1_1 = StateVector([1, 1])

    sv1 = sv_0_1 + sv_1_0
    sv1_bis = 1.0000001*sv_0_1 + 0.9999999*sv_1_0
    sv2 = sv_0_1 - sv_1_0
    sv3 = sv_0_1 + sv_1_1
    sv4 = sv_0_1

    assert_sv_close(sv1, sv1_bis)

    with pytest.raises(AssertionError):
        assert_sv_close(sv1, sv2)
    with pytest.raises(AssertionError):
        assert_sv_close(sv1, sv3)
    with pytest.raises(AssertionError):
        assert_sv_close(sv1, sv4)
    with pytest.raises(AssertionError):
        assert_sv_close(sv4, sv1)
