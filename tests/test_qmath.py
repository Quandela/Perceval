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

import itertools
import pytest
import time

from perceval import BasicState, StateVector, BSDistribution, SVDistribution
from perceval.utils.qmath import exponentiation_by_squaring, distinct_permutations
from _test_utils import assert_sv_close, assert_svd_close


def test_exponentiation():
    with pytest.raises(ValueError):
        exponentiation_by_squaring(12, 0)
    # Numbers
    assert exponentiation_by_squaring(12, 1), 12
    assert exponentiation_by_squaring(8, 2), 64
    assert exponentiation_by_squaring(52, 13), 20325604337285010030592

    # Basic state
    bs = BasicState("|1,0,1,0,0>")
    assert bs**3 == bs * bs * bs

    # Annoted basic state
    annot_bs = BasicState("|0,0,{_:0}{_:1},0>")
    assert annot_bs**5 == annot_bs * annot_bs * annot_bs * annot_bs * annot_bs

    # State vector
    sv = StateVector(BasicState("|1,0,4,0,0,2,0,1>")) - 1j * StateVector(BasicState("|1,0,3,0,2,1,0,1>"))
    assert_sv_close(sv, sv)
    assert_sv_close(sv**2, sv * sv)
    assert_sv_close(sv**7, sv * sv * sv * sv * sv * sv * sv)

    # Basic state Distribution
    bsd = BSDistribution()
    bsd[BasicState([0, 0])] = 0.25
    bsd[BasicState([1, 0])] = 0.25
    bsd[BasicState([0, 1])] = 0.25
    bsd[BasicState([2, 0])] = 0.125
    bsd[BasicState([0, 2])] = 0.125
    assert bsd**2 == bsd * bsd
    assert bsd**7 == bsd * bsd * bsd * bsd * bsd * bsd * bsd

    # State vector Distribution
    svd = SVDistribution()
    svd[StateVector([0]) + StateVector([1])] = 0.25
    svd[StateVector([0]) + 1j*StateVector([1])] = 0.35
    svd[StateVector([1])] = 0.4
    assert_svd_close(svd, svd)
    assert_svd_close(svd**2, svd * svd)
    assert_svd_close(svd**5, svd * svd * svd * svd * svd)


@pytest.mark.parametrize("parameters", [('mississippi', 0),
                                        ('mississippi', 1),
                                        ('mississippi', 6),
                                        ('mississippi', 7),
                                        ('mississippi', 12),
                                        ([0, 1, 1, 0], 0),
                                        ([0, 1, 1, 0], 1),
                                        ([0, 1, 1, 0], 2),
                                        ([0, 1, 1, 0], 3),
                                        ([0, 1, 1, 0], 4),
                                        ([0, 1, 1, 0], None),
                                        (['a'], 0),
                                        (['a'], 1),
                                        (['a'], 5),
                                        ([], 0),
                                        ([], 1),
                                        ([], 4),])
def test_distinct_permutations(parameters):
    iterable, r = parameters
    assert sorted(set(itertools.permutations(iterable, r))) == sorted(distinct_permutations(iter(iterable), r))
