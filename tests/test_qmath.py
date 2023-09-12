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

from perceval import BasicState, StateVector, BSDistribution, SVDistribution
from perceval.utils.qmath import exponentiation_by_squaring
from pytest import approx
from time import time


def almost_equal_sv_distribution(lhsvd, rhsvd):
    lhkeys, rhkeys = [[{svkeys for svkeys in svdkeys.keys()} for svdkeys in svd.keys()] for svd in [lhsvd, rhsvd]]
    # check if all the basic state are here
    assert lhkeys == rhkeys
    # search correct key in both dict (since the keys can be different because of pointing float error causing different probabilities)
    for incomplete_key in lhkeys:
        found_key = False
        for lkey in lhsvd.keys():
            found_key = True
            for bs in incomplete_key:
                if bs not in lkey:
                    found_key = False
                    break
            if found_key:
                break
        found_key = False
        for rkey in rhsvd.keys():
            found_key = True
            for bs in incomplete_key:
                if bs not in rkey:
                    found_key = False
                    break
            if found_key:
                break
        # compare the found keys
        almost_equal_state_vector(lkey, rkey)
        assert lhsvd[lkey] == approx(rhsvd[rkey])


def almost_equal_state_vector(lhsv, rhsv):
    assert set(lhsv.keys()) == set(rhsv.keys())
    for key in lhsv.keys():
        assert lhsv[key] == approx(rhsv[key])  # approx <=> error < 1e-6


def test_exponentiation():
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
    almost_equal_state_vector(sv, sv)
    almost_equal_state_vector(sv**2, sv * sv)
    almost_equal_state_vector(sv**7, sv * sv * sv * sv * sv * sv * sv)

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
    svd[StateVector([1, 0]) + StateVector([0, 1])] = 0.3
    svd[StateVector([1, 1]) + 1j * StateVector([0, 1])] = 0.3
    svd[StateVector('|2,0>') + StateVector([1, 1])] = 0.4
    almost_equal_sv_distribution(svd, svd)
    almost_equal_sv_distribution(svd**2, svd * svd)
    almost_equal_sv_distribution(svd**5, svd * svd * svd * svd * svd)
