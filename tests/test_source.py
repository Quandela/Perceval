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
from collections import defaultdict

import pytest
import math

from perceval import BSDistribution, StateVector, filter_distribution_photon_count
from perceval.components import Source
from perceval.utils import BasicState
from perceval.utils.conversion import samples_to_probs
from perceval.utils.dist_metrics import tvd_dist
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


def test_source_sample():
    nb_samples = 200

    bs = BasicState("|1,1>")
    source_1 = Source(emission_probability=0.9, multiphoton_component=0.1, losses=0.1, indistinguishability=0.9)
    source_2 = Source(emission_probability=0.9, multiphoton_component=0.1, losses=0.1, indistinguishability=0.9)

    # generate samples directly from the source
    samples_from_source = source_1.generate_samples(nb_samples, bs)
    assert len(samples_from_source) == nb_samples

    dist_samples = samples_to_probs(samples_from_source)

    # compare these samples with complete distribution
    dist = source_2.generate_distribution(bs,0)
    dist = BSDistribution({str(key):value for key,value in dist.items()}) # change SVD to BSD

    for el in set(dist.keys()) - set(dist_samples.keys()):
        dist_samples[el] = 0

    tvd = tvd_dist(dist_samples, dist)
    assert tvd == pytest.approx(0.0, abs=0.15)  # total variation between two distributions is less than 0.15

    # number of photons in samples
    nb_1p = 0
    nb_2p = 0
    nb_3p = 0
    for sample in samples_from_source:
        if sample.n == 1:
            nb_1p += 1
        elif sample.n == 2:
            nb_2p += 1
        elif sample.n == 3:
            nb_3p += 1
    assert nb_2p > nb_1p
    assert nb_2p > nb_3p
    assert nb_1p > nb_3p


def test_source_table():
    brightness = .875
    g2 = .25  # Values chosen so probabilities are "round" values
    hom = 1
    losses = 0.1

    s = Source(brightness, g2, hom, losses)

    # Works because there is no HOM.
    # Doesn't work with HOM as we sum up some g2 and HOM in the same annotations in source.generate_distribution
    def bsd_to_prob_table(bsd) -> dict:
        res = defaultdict(float)

        for bs, p in bsd.items():
            signal = 0
            g2_alone = 0
            g2_signal = 0

            for m in range(bs.m):
                trunc = bs[m:m + 1]  # keeps the annotations

                if trunc.n == 1:
                    val = int(trunc.get_photon_annotation(0).str_value("_")) if bs.has_annotations else 0
                    if val == 0:
                        signal += 1
                    else:
                        g2_alone += 1

                elif trunc.n == 2:
                    g2_signal += 1

            res[(signal, g2_alone, g2_signal)] += p

        return res

    true_svd = s.generate_distribution(BasicState([1, 1]))
    true_bsd = BSDistribution({sv[0]: p for sv, p in true_svd.items()})

    prob_table, phys_perf, zpp = s._compute_prob_table(2)

    assert prob_table == pytest.approx(bsd_to_prob_table(true_bsd))
    assert phys_perf == pytest.approx(1)
    assert zpp == pytest.approx(true_svd[StateVector([0, 0])])

    truncated_svd, perf = filter_distribution_photon_count(true_bsd, 2)

    prob_table, phys_perf, zpp = s._compute_prob_table(2, 2)
    assert prob_table == pytest.approx(bsd_to_prob_table(truncated_svd))
    assert phys_perf == pytest.approx(perf)
    assert zpp == pytest.approx(true_svd[StateVector([0, 0])])
