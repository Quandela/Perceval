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
from perceval import BasicState, StateVector, tensorproduct

sv0 = StateVector([0, 1]) + StateVector([1, 1]) * 1j
sv1 = StateVector([2, 3]) + StateVector([4, 5])
bs = BasicState([6, 7])


def test_mul():
    assert sv0 * sv1 == pytest.approx(0.5 * StateVector([0, 1, 2, 3])
                                      + 0.5j * StateVector([1, 1, 2, 3])
                                      + 0.5 * StateVector([0, 1, 4, 5])
                                      + 0.5j * StateVector([1, 1, 4, 5])), "SV multiplication's wrong"

    assert sv0 * bs == pytest.approx(0.5 * 2 ** 0.5 * StateVector([0, 1, 6, 7])
                                     + 0.5 * 2 ** 0.5 * 1j * StateVector(
        [1, 1, 6, 7])), "SV with BS multiplication's wrong"

    assert bs * sv0 == pytest.approx(0.5 * 2 ** 0.5 * StateVector([6, 7, 0, 1])
                                     + 0.5 * 2 ** 0.5 * 1j * StateVector(
        [6, 7, 1, 1])), "BS with SV multiplication's wrong"


def test_tensorproduct():
    assert tensorproduct([sv0, sv1, bs]) == pytest.approx(sv0 * sv1 * bs), "tensor product's wrong"


def test_power():
    power = 5
    sv_list = power * [sv0]

    assert sv0 ** power == pytest.approx(tensorproduct(sv_list)), "SV pow is wrong"
    assert bs ** power == pytest.approx(BasicState([6, 7] * power)), "BS pow is wrong"
