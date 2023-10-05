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
import math

from perceval.components import Source
from perceval.rendering.pdisplay import pdisplay_state_distrib
from _test_utils import strip_line_12, assert_svd_close, dict2svd


def test_tag():
    s = Source()
    assert s.get_tag("discernability_tag", False) == 0
    assert s.get_tag("discernability_tag", True) == 1
    assert s.get_tag("discernability_tag", False) == 1


def test_intermediate_probs():
    assert pytest.approx((1, 0, 0)) == Source()._get_probs()
    p1 = .8
    p2 = .01
    beta = p1 + p2
    g2 = 2 * p2 / (p1 + 2 * p2) ** 2
    s = Source(emission_probability=beta, multiphoton_component=g2, losses=.1)
    assert pytest.approx((.72, .0009, .0081)) == s._get_probs()


def test_source_pure():
    s = Source()
    svd = s.probability_distribution()
    assert strip_line_12(pdisplay_state_distrib(svd)) == strip_line_12("""
            +-------+-------------+
            | state | probability |
            +-------+-------------+
            |  |1>  |      1      |
            +-------+-------------+""")
    assert_svd_close(svd, dict2svd({"|1>": 1}))


def test_source_emission():
    s = Source(emission_probability=0.4)
    svd = s.probability_distribution()
    assert_svd_close(svd, dict2svd({"|0>": 0.6, "|1>": 0.4}))


def test_source_emission_g2():
    s = Source(emission_probability=0.4, multiphoton_component=0.1, multiphoton_model="indistinguishable")
    svd = s.probability_distribution()
    assert_svd_close(svd, dict2svd({"|0>": 3/5, "|2>": 0.008336953374561418, "|1>": 0.391663}))


def test_source_emission_g2_losses_indistinguishable():
    s = Source(emission_probability=0.4, multiphoton_component=0.1, losses=0.08, indistinguishability=0.9,
               multiphoton_model="indistinguishable")
    svd = s.probability_distribution()
    assert_svd_close(svd, dict2svd({"|0>": 0.631386, "|2{_:0}>": 0.006694286297288383, "|{_:0}{_:1}>": 3.62111e-4,
                                    "|{_:0}>": 0.343035, "|{_:1}>": 0.01852243527847053}))


def test_source_indistinguishability():
    s = Source(indistinguishability=0.5)
    svd = s.probability_distribution()
    assert len(svd) == 2
    for k, v in svd.items():
        assert len(k) == 1
        state = k[0]
        assert state.n == 1
        annot = state.get_photon_annotation(0)
        assert "_" in annot, "missing distinguishability feature _"
        if annot["_"] != 0:
            assert pytest.approx(1-math.sqrt(0.5)) == v
        else:
            assert pytest.approx(math.sqrt(0.5)) == v


def test_source_multiple_photons_per_mode():
    s = Source()
    for nphotons in range(2,10):
        svd = s.probability_distribution(nphotons)
        assert_svd_close(svd, dict2svd({f"|{nphotons}>": 1}))

    ep = 0.41
    s = Source(emission_probability=ep)
    svd = s.probability_distribution(2)
    assert_svd_close(svd, dict2svd({"|0>": (1-ep)**2, "|1>": ep*(1-ep)*2, "|2>": ep**2}))
