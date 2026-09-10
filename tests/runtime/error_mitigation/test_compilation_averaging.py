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
from exqalibur import FockState

from ..._test_utils import assert_bsd_close
from perceval import CompilationAveraging, Computation, CommandFactory, Experiment, NoiseModel, Command, Imperfections, \
    BSDistribution
from perceval.utils.constants import KEY_SHOTS_USED


def test_computation_extension():
    with pytest.raises(AssertionError):
        CompilationAveraging(0)

    averaging = CompilationAveraging(3)

    # Test without "compilation_seed" in signature
    computation = Computation(CommandFactory.probs, Experiment())
    imperfections = Imperfections(NoiseModel(), [])

    comp_list = averaging.extend_computation(computation, imperfections)
    assert len(comp_list) == 1

    # Test with "compilation_seed" in signature
    command = Command("sample_count",
                      [("max_shots", int, True), ("max_samples", int, False), ("compilation_seed", int, False)],
                      apply_emt=True)

    computation = Computation(command, Experiment())
    computation.add_params(max_shots = 50000, max_samples = 10000)

    comp_list = averaging.extend_computation(computation, imperfections)
    assert len(comp_list) == 3
    assert all(comp.command.name == "probs" for comp in comp_list)
    assert sum(comp.parameters["max_shots"] for comp in comp_list) == 50000
    assert sum(comp.parameters["max_samples"] for comp in comp_list) == 10000

    seeds = {comp.parameters["compilation_seed"] for comp in comp_list}
    assert len(seeds) == 3

    computation.add_params(max_shots = 60000, max_samples = 15000)  # So the numbers are divisible by the number of repetitions
    comp_list = averaging.extend_computation(computation, imperfections)
    assert len(comp_list) == 3
    assert all(comp.parameters["max_shots"] == 20000 for comp in comp_list)
    assert all(comp.parameters["max_samples"] == 5000 for comp in comp_list)


def prepare_test():
    sub_results = [
        {"results": BSDistribution({
            FockState([1, 1]): 0.4,
            FockState([2, 1]): 0.1,
            FockState([3, 1]): 0.5,
        }),
            "global_perf": 0.3,
            "physical_perf": 0.75,
            "logical_perf": 0.4,
            KEY_SHOTS_USED: 2500},

        {"results": BSDistribution({
            FockState([1, 1]): 0.1,
            FockState([2, 1]): 0.2,
            FockState([3, 1]): 0.7,
        }),
            "global_perf": 0.2,
            "physical_perf": 0.25,
            "logical_perf": 0.8,
            KEY_SHOTS_USED: 3500},

        {"results": BSDistribution({
            FockState([1, 1]): 0.6,
            FockState([2, 1]): 0.1,
            FockState([3, 1]): 0.3,
        }),
            "global_perf": 0.3,
            "physical_perf": 0.5,
            "logical_perf": 0.6,
            KEY_SHOTS_USED: 4000}
    ]

    expected = {"results": BSDistribution({
        FockState([1, 1]): (0.4 + 0.1 + 0.6) / 3,
        FockState([2, 1]): (0.1 + 0.2 + 0.1) / 3,
        FockState([3, 1]): (0.5 + 0.7 + 0.3) / 3,
    }),
        "physical_perf": (0.75 + 0.25 + 0.5) / 3,
        "logical_perf": (0.4 + 0.8 + 0.6) / 3,
        "global_perf": (0.3 + 0.2 + 0.3) / 3,
        KEY_SHOTS_USED: 10000}

    return expected, sub_results


def test_recombination():
    expected, sub_results = prepare_test()

    averaging = CompilationAveraging(3)

    computation = Computation(CommandFactory.probs, Experiment())
    computation.add_params(max_samples = expected[KEY_SHOTS_USED])

    res = averaging.parse_results(computation, sub_results, Imperfections(NoiseModel(), []))

    bsd = res.pop("results")
    bsd_expected = expected.pop("results")
    assert res == pytest.approx(expected)
    assert_bsd_close(bsd, bsd_expected)
