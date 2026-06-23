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
from exqalibur import BSCount, BSSamples

from perceval import AbstractMitigation, Computation, NoiseModel, CommandFactory, Experiment, BSDistribution, FockState
from tests._test_utils import assert_bsd_close


class DummyMitigation(AbstractMitigation):

    def extend_computation(self, computation: Computation, noise: NoiseModel) -> list[Computation]:
        return [computation]

    def _parse_results(self, computation: Computation, results: list[dict], noise) -> dict:
        return results[0]


def test_min_photon_filter():
    mitigation = DummyMitigation()

    e = Experiment()
    e.min_detected_photons_filter(2)

    top_layer = Computation(CommandFactory.probs, e)
    sub_results = [
        {"results": BSDistribution({
            FockState("|1, 0>"): 0.4,  # Will be rejected
            FockState("|1, 1>"): 0.6,
        }),
        "global_perf": 0.7,
        "physical_perf": 0.7,
        "logical_perf": 1.,}
    ]

    res = mitigation.parse_results(top_layer, sub_results, NoiseModel())
    expected = BSDistribution({FockState("|1, 1>"): 1.})

    assert_bsd_close(res["results"], expected)
    assert res["global_perf"] == pytest.approx(0.7 * 0.6)
    assert res["physical_perf"] == pytest.approx(0.7 * 0.6)
    assert res["logical_perf"] == pytest.approx(1.)

    top_layer = Computation(CommandFactory.sample_count, e)
    sub_results = [
        {"results": BSCount({
            FockState("|1, 0>"): 200,  # Will be rejected
            FockState("|1, 1>"): 300,
        }),
        "global_perf": 0.7}
    ]

    res = mitigation.parse_results(top_layer, sub_results, NoiseModel())
    expected = BSCount({FockState("|1, 1>"): 300})

    assert res["results"] == expected
    assert res["global_perf"] == pytest.approx(0.7 * 0.6)

    top_layer = Computation(CommandFactory.samples, e)
    sub_results = [
        {"results": BSSamples([FockState("|1, 0>"), FockState("|1, 1>"), FockState("|1, 1>"), FockState("|1, 0>"), FockState("|1, 1>")]),
        "global_perf": 0.7}
    ]

    res = mitigation.parse_results(top_layer, sub_results, NoiseModel())
    expected = BSSamples([FockState("|1, 1>"), FockState("|1, 1>"), FockState("|1, 1>")])

    assert res["results"] == expected
    assert res["global_perf"] == pytest.approx(0.7 * 0.6)


def test_logical_postprocess():
    # Same test as above, but uses logical selection instead
    mitigation = DummyMitigation()

    e = Experiment()
    e.set_postselection("[0,1] == 2")

    top_layer = Computation(CommandFactory.probs, e)
    sub_results = [
        {"results": BSDistribution({
            FockState("|1, 0>"): 0.4,  # Will be rejected
            FockState("|1, 1>"): 0.6,
        }),
        "global_perf": 0.7,
        "physical_perf": 0.7,
        "logical_perf": 1., }
    ]

    res = mitigation.parse_results(top_layer, sub_results, NoiseModel())
    expected = BSDistribution({FockState("|1, 1>"): 1.})

    assert_bsd_close(res["results"], expected)
    assert res["global_perf"] == pytest.approx(0.7 * 0.6)
    assert res["physical_perf"] == pytest.approx(0.7)
    assert res["logical_perf"] == pytest.approx(0.6)

    top_layer = Computation(CommandFactory.sample_count, e)
    sub_results = [
        {"results": BSCount({
            FockState("|1, 0>"): 200,  # Will be rejected
            FockState("|1, 1>"): 300,
        }),
        "global_perf": 0.7}
    ]

    res = mitigation.parse_results(top_layer, sub_results, NoiseModel())
    expected = BSCount({FockState("|1, 1>"): 300})

    assert res["results"] == expected
    assert res["global_perf"] == pytest.approx(0.7 * 0.6)

    top_layer = Computation(CommandFactory.samples, e)
    sub_results = [
        {"results": BSSamples(
            [FockState("|1, 0>"), FockState("|1, 1>"), FockState("|1, 1>"), FockState("|1, 0>"), FockState("|1, 1>")]),
         "global_perf": 0.7}
    ]

    res = mitigation.parse_results(top_layer, sub_results, NoiseModel())
    expected = BSSamples([FockState("|1, 1>"), FockState("|1, 1>"), FockState("|1, 1>")])

    assert res["results"] == expected
    assert res["global_perf"] == pytest.approx(0.7 * 0.6)


def test_automatic_conversion_from_bsc():
    mitigation = DummyMitigation()
    e = Experiment()

    # The layer above asked for a BSD, but I work with BSC
    top_layer = Computation(CommandFactory.probs, e)
    top_layer.add_params(10000)
    sub_results = [
        {"results": BSCount({
            FockState("|1, 0>"): 2,
            FockState("|1, 1>"): 3,
        }),
        "global_perf": 0.7}
    ]

    res = mitigation.parse_results(top_layer, sub_results, NoiseModel())
    expected = BSDistribution({FockState("|1, 0>"): 0.4, FockState("|1, 1>"): 0.6})

    assert_bsd_close(res["results"], expected)

    # TODO: make BSCount to BSSamples draw the exact same number of samples (PCVL-1251)
    # Now the layer above asked for BSS
    # top_layer = Computation(CommandFactory.samples, e)
    # top_layer.add_params(5)
    # res = mitigation.parse_results(top_layer, sub_results)
    #
    # assert isinstance(res["results"], BSSamples)
    # assert len(res["results"]) == 5
    # bsd_res = BSDistribution()
    # for state in res["results"]:
    #     bsd_res[state] += 1
    #
    # bsd_res.normalize()
    # assert bsd_res == expected

def test_automatic_conversion_from_bsd():
    mitigation = DummyMitigation()
    e = Experiment()

    # The layer above asked for a BSC, but I work with BSD
    top_layer = Computation(CommandFactory.sample_count, e)
    top_layer.add_params(10000)
    sub_results = [
        {"results": BSDistribution({
            FockState("|1, 0>"): 0.4,
            FockState("|1, 1>"): 0.6,
        }),
        "global_perf": 0.7}
    ]

    res = mitigation.parse_results(top_layer, sub_results, NoiseModel())
    assert isinstance(res["results"], BSCount)
    assert res["results"].total() == 10000

    # Now the layer above asked for BSS
    top_layer = Computation(CommandFactory.samples, e)
    top_layer.add_params(10000)
    res = mitigation.parse_results(top_layer, sub_results, NoiseModel())

    assert isinstance(res["results"], BSSamples)
    assert len(res["results"]) == 10000


def test_automatic_conversion_from_bss():
    mitigation = DummyMitigation()
    e = Experiment()

    # The layer above asked for a BSC, but I work with BSS
    top_layer = Computation(CommandFactory.sample_count, e)
    sub_results = [
        {"results": BSSamples(
            [FockState("|1, 0>"), FockState("|1, 1>"), FockState("|1, 1>"), FockState("|1, 0>"), FockState("|1, 1>")]),
        "global_perf": 0.7}
    ]

    res = mitigation.parse_results(top_layer, sub_results, NoiseModel())
    assert isinstance(res["results"], BSCount)
    assert res["results"].total() == 5
    assert res["results"][FockState("|1, 0>")] == 2
    assert res["results"][FockState("|1, 1>")] == 3

    # Now the layer above asked for BSD
    top_layer = Computation(CommandFactory.probs, e)
    res = mitigation.parse_results(top_layer, sub_results, NoiseModel())

    assert isinstance(res["results"], BSDistribution)

    expected = BSDistribution({
            FockState("|1, 0>"): 0.4,
            FockState("|1, 1>"): 0.6,
        })

    assert_bsd_close(res["results"], expected)
