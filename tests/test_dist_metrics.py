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

from perceval.utils.dist_metrics import tvd_dist, kl_divergence
from perceval.utils import BSDistribution, BasicState
from copy import copy


bs1 = BasicState([0, 1, 0])
bs2 = BasicState([0, 0, 0])
bs3 = BasicState([0, 0, 1])
bs4 = BasicState([1, 0, 0])

def test_tvd_identical_dist():
    target_bsd = BSDistribution()
    target_bsd[bs1] = 0.5
    target_bsd[bs2] = 0.5

    bsd_to_comp = copy(target_bsd)

    assert tvd_dist(target_bsd, bsd_to_comp) == 0


def test_tvd_disjoint_dist():
    target_bsd = BSDistribution()
    target_bsd[bs1] = 0.5
    target_bsd[bs2] = 0.5

    bsd_to_comp = BSDistribution()
    bsd_to_comp[bs3] = 1
    bsd_to_comp[bs4] = 0

    assert tvd_dist(target_bsd, bsd_to_comp) == 1.0


def test_tvd_one_empty_dist():
    target_bsd = BSDistribution()
    target_bsd[bs1] = 0.3
    target_bsd[bs2] = 0.7

    assert tvd_dist(target_bsd, BSDistribution()) == 0.5


def test_kl_div_identical_dist():
    target_bsd = BSDistribution()
    target_bsd[bs1] = 0.5
    target_bsd[bs2] = 0.5

    bsd_to_comp = copy(target_bsd)

    assert kl_divergence(target_bsd, bsd_to_comp) == 0


def test_kl_div_unequal_dist():
    ideal_bsd = BSDistribution()  # Binomial
    ideal_bsd[bs1] = 9/25
    ideal_bsd[bs2] = 12/25
    ideal_bsd[bs3] = 4/25

    model_bsd = BSDistribution()  # uniform
    model_bsd[bs1] = 1/3
    model_bsd[bs2] = 1/3
    model_bsd[bs3] = 1/3

    assert kl_divergence(ideal_bsd, model_bsd) != kl_divergence(model_bsd, ideal_bsd)  # it is not symmetric

    model2_bsd = BSDistribution()  # Binomial
    model2_bsd[bs1] = 8/25
    model2_bsd[bs2] = 10/25
    model2_bsd[bs3] = 7/25

    assert kl_divergence(ideal_bsd, model_bsd) > kl_divergence(ideal_bsd, model2_bsd)  # Model 2 closer to the ideal
