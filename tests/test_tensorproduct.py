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
from perceval import BasicState, StateVector, tensorproduct, BSDistribution, SVDistribution
from unittest.mock import patch
from perceval.utils.logging import ExqaliburLogger
from _test_utils import LogChecker

sv0 = StateVector([0, 1]) + StateVector([1, 1]) * 1j
sv1 = StateVector([2, 3]) + StateVector([4, 5])
bs = BasicState([6, 7])


def _assert_sv_approx_eq(sv1: StateVector, sv2: StateVector, error_msg="Assertion error"):
    sv1.normalize()
    sv2.normalize()
    for state in sv1.keys():
        assert state in sv2, error_msg
        assert sv1[state] == pytest.approx(sv2[state]), error_msg


def test_mul():
    result = sv0 * sv1
    expected = (0.5 * StateVector([0, 1, 2, 3])
                + 0.5j * StateVector([1, 1, 2, 3])
                + 0.5 * StateVector([0, 1, 4, 5])
                + 0.5j * StateVector([1, 1, 4, 5]))
    _assert_sv_approx_eq(result, expected, "SV multiplication is wrong")

    result = sv0 * bs
    expected = (0.5 * 2 ** 0.5 * StateVector([0, 1, 6, 7])
                + 0.5 * 2 ** 0.5 * 1j * StateVector([1, 1, 6, 7]))
    _assert_sv_approx_eq(result, expected, "SV with BS multiplication is wrong")

    result = StateVector(bs) * sv0
    expected = (0.5 * 2 ** 0.5 * StateVector([6, 7, 0, 1])
                + 0.5 * 2 ** 0.5 * 1j * StateVector([6, 7, 1, 1]))
    _assert_sv_approx_eq(result, expected, "BS with SV multiplication is wrong")


def test_tensorproduct():
    result = tensorproduct([sv0, sv1, bs])
    expected = sv0 * sv1 * bs
    _assert_sv_approx_eq(result, expected, "tensor product is wrong")


def test_power():
    power = 5
    sv_list = power * [sv0]

    result = sv0 ** power
    expected = tensorproduct(sv_list)
    _assert_sv_approx_eq(result, expected, "SV pow is wrong")

    result = bs ** power
    expected = BasicState([6, 7] * power)
    assert result == expected, "BS pow is wrong"


@patch.object(ExqaliburLogger, "warn")
def test_bsd_tensor_product(mock_warn):
    bsd_1 = BSDistribution({BasicState([2, 3]): .4,
                            BasicState([0, 1]): .6})

    bsd_2 = BSDistribution({BasicState([4, 5]): .3,
                            BasicState([6, 7]): .7})

    bsd_3 = BSDistribution({BasicState([8, 9]): .3,
                            BasicState([10, 11]): .5,
                            BasicState([12, 13]): .2})

    assert BSDistribution.tensor_product(bsd_1, bsd_2) == pytest.approx(bsd_1 * bsd_2), "BSD tensor product is wrong"

    assert BSDistribution.tensor_product(bsd_1, bsd_2, merge_modes=True) == pytest.approx(BSDistribution({
        BasicState([6, 8]): .4 * .3 + .6 * .7,
        BasicState([8, 10]): .4 * .7,
        BasicState([4, 6]): .6 * .3,
    })), "Wrong tensor product result when merge_modes is True"

    assert (BSDistribution.tensor_product(bsd_1, bsd_2, merge_modes=True)
            == pytest.approx(BSDistribution.tensor_product(bsd_2, bsd_1, merge_modes=True))), \
        "BSD tensor product is not symmetric when merge_modes is True"

    assert BSDistribution.list_tensor_product([bsd_1, bsd_2, bsd_3]) == pytest.approx(bsd_1 * bsd_2 * bsd_3), \
        "BSD list tensor product is wrong"

    product = BSDistribution()
    for bsd in [bsd_1, bsd_2, bsd_3]:
        product = BSDistribution.tensor_product(product, bsd, merge_modes=True)

    assert BSDistribution.list_tensor_product([bsd_1, bsd_2, bsd_3], merge_modes=True) == pytest.approx(product), \
        "Wrong list tensor product result when merge_modes is True"

    # Now with empty BSD
    bsd_list = [bsd_1, bsd_2, BSDistribution(), bsd_3]

    with LogChecker(mock_warn):
        assert BSDistribution.list_tensor_product(bsd_list, merge_modes=True) == pytest.approx(product), \
            "Wrong list tensor product result when merge_modes is True and there are empty BSD"


@patch.object(ExqaliburLogger, "warn")
def test_svd_tensor_product(mock_warn):
    svd_1 = SVDistribution({StateVector([2, 3]): .4,
                            StateVector([0, 1]): .6})

    svd_2 = SVDistribution({StateVector([4, 5]): .3,
                            StateVector([6, 7]): .7})

    svd_3 = SVDistribution({StateVector([8, 9]): .3,
                            StateVector([10, 11]): .5,
                            StateVector([12, 13]): .2})

    assert SVDistribution.tensor_product(svd_1, svd_2) == pytest.approx(svd_1 * svd_2), "SVD tensor product is wrong"

    assert SVDistribution.list_tensor_product([svd_1, svd_2, svd_3]) == pytest.approx(svd_1 * svd_2 * svd_3), \
        "SVD list tensor product is wrong"

    # Now with empty SVD
    bsd_list = [svd_1, svd_2, SVDistribution(), svd_3]

    with LogChecker(mock_warn):
        assert SVDistribution.list_tensor_product(bsd_list) == pytest.approx(svd_1 * svd_2 * svd_3), \
            "Wrong list tensor product result when there are empty BSD"
