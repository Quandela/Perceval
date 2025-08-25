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
from collections import Counter

from perceval import filter_distribution_photon_count, SVDistribution, \
    anonymize_annotations, FockState, NoisyFockState, StateVector
from perceval.components import Source
from perceval.utils.conversion import samples_to_probs
from perceval.utils.dist_metrics import tvd_dist
from perceval.rendering.pdisplay import pdisplay_state_distrib
from _test_utils import strip_line_12, assert_svd_close, dict2svd


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
    assert_svd_close(svd, dict2svd({"|0>": 0.631386, "|2{0}>": 0.006694286297288383, "|{0}{1}>": 3.62111e-4,
                                    "|{0}>": 0.343035, "|{1}>": 0.01852243527847053}))


def test_source_indistinguishability():
    s = Source(indistinguishability=0.5)
    svd = s.probability_distribution()
    assert len(svd) == 2
    for k, v in svd.items():
        assert len(k) == 1
        state = k[0]
        assert state.n == 1
        assert isinstance(state, NoisyFockState)
        if str(state) != "|{0}>":
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


def test_source_sample_no_filter():
    nb_samples = 200

    bs = FockState("|1,1>")
    source_1 = Source(emission_probability=0.9, multiphoton_component=0.1, losses=0.1, indistinguishability=0.9)
    source_2 = Source(emission_probability=0.9, multiphoton_component=0.1, losses=0.1, indistinguishability=0.9)
    source_2.simplify_distribution = True

    # generate samples directly from the source
    samples_from_source = source_1.generate_samples(nb_samples, bs)
    assert len(samples_from_source) == nb_samples

    counter_samples = Counter(samples_from_source)
    total = counter_samples.total()
    dist_samples = SVDistribution()
    for k, v in counter_samples.items():
        dist_samples.add(StateVector(k), v / total)

    # compare these samples with complete distribution
    dist_raw = source_2.generate_distribution(bs,0)
    dist = SVDistribution()
    for k, v in dist_raw.items():
        dist.add(StateVector(NoisyFockState(k[0])), v)

    # just avoid the warning in tvd_dist
    for el in set(dist_samples.keys()) - set(dist.keys()):
        dist[el] = 0
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


@pytest.mark.parametrize("brightness", [1, 0.7])
@pytest.mark.parametrize("g2", [0, 0.3])
@pytest.mark.parametrize("hom", [1, 0.6, 0])
@pytest.mark.parametrize("losses", [0, 0.4])
@pytest.mark.parametrize("multiphoton_model", ['distinguishable', 'indistinguishable'])
def test_source_samples_with_filter(brightness, g2, hom, losses, multiphoton_model):
    nb_samples = 200
    min_detected_photons = 2

    bs = FockState("|1,1>")
    source_1 = Source(brightness, g2, hom, losses)
    source_2 = Source(brightness, g2, hom, losses)

    # generate samples directly from the source
    samples_from_source = source_1.generate_samples(nb_samples, bs, min_detected_photons)
    assert len(samples_from_source) == nb_samples
    assert all(bs.n >= min_detected_photons for bs in samples_from_source)

    counter_samples = Counter(samples_from_source)
    total = counter_samples.total()
    dist_samples = SVDistribution()
    for k, v in counter_samples.items():
        dist_samples.add(StateVector(k), v / total)
    dist_samples = anonymize_annotations(dist_samples, annot_tag="_")  # to be able to compare the distributions

    # compare these samples with complete distribution
    source_2.simplify_distribution = True
    dist = source_2.generate_distribution(bs, 0)
    dist_raw = filter_distribution_photon_count(dist, min_detected_photons)[0]
    dist = SVDistribution()
    for k, v in dist_raw.items():
        dist.add(StateVector(NoisyFockState(k[0])), v)

    # just avoid the warning in tvd_dist
    for el in set(dist_samples.keys()) - set(dist.keys()):
        dist[el] = 0
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

    assert nb_1p == 0
    assert nb_2p > nb_3p
