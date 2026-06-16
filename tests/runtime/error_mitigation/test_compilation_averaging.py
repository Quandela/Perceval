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

import random

import pytest
from exqalibur import FockState
from exqalibur.exqalibur import PostSelect

from perceval import CompilationAveraging, Computation, CommandFactory, Experiment, NoiseModel, Command, BSCount, \
    apply_min_photons, apply_post_select
from perceval.utils.constants import KEY_SHOTS_USED


def test_computation_extension():
    with pytest.raises(AssertionError):
        CompilationAveraging(0)

    averaging = CompilationAveraging(3)

    # Test without "compilation_seed" in signature
    computation = Computation(CommandFactory.probs, Experiment())

    comp_list = averaging.extend_computation(computation, NoiseModel())
    assert len(comp_list) == 1

    # Test with "compilation_seed" in signature
    command = Command("probs", [("max_shots", int, True), ("max_samples", int, False), ("compilation_seed", int, False)], apply_emt=True)

    computation = Computation(command, Experiment())
    computation.add_params(max_shots = 50000, max_samples = 10000)

    comp_list = averaging.extend_computation(computation, NoiseModel())
    assert len(comp_list) == 3
    assert all(comp.command.name == "sample_count" for comp in comp_list)
    assert sum(comp.parameters["max_shots"] for comp in comp_list) == 50000
    assert sum(comp.parameters["max_samples"] for comp in comp_list) == 10000

    seeds = {comp.parameters["compilation_seed"] for comp in comp_list}
    assert len(seeds) == 3

    computation.add_params(max_shots = 60000, max_samples = 15000)  # So the numbers are divisible by the number of repetitions
    comp_list = averaging.extend_computation(computation, NoiseModel())
    assert len(comp_list) == 3
    assert all(comp.parameters["max_shots"] == 20000 for comp in comp_list)
    assert all(comp.parameters["max_samples"] == 5000 for comp in comp_list)


def prepare_test():
    raw_results = BSCount({FockState([0, 1]): 16,
                           FockState([1, 0]): 32,
                           FockState([1, 1]): 64,
                           FockState([2, 0]): 128,
                           FockState([0, 2]): 256})

    # Distributes to sub-results
    raw_sub_results = [BSCount(), BSCount(), BSCount()]

    for state, count in raw_results.items():
        for _ in range(count):
            raw_sub_results[random.randint(0, len(raw_sub_results) - 1)][state] += 1

    min_photons = 2
    post_select = PostSelect("[1] >= 1")
    heralds = {}

    sub_results = []
    for i, bsc in enumerate(raw_sub_results):
        shots_used = bsc.total()
        bsc, phys_perf = apply_min_photons(bsc, min_photons)
        bsc, log_perf = apply_post_select(bsc, post_select, heralds, True)
        sub_results.append({"results": bsc,
                            "physical_perf": phys_perf,
                            "logical_perf": log_perf,
                            "global_perf": phys_perf * log_perf,
                            KEY_SHOTS_USED: shots_used})

    shots_used = raw_results.total()
    raw_results, phys_perf = apply_min_photons(raw_results, min_photons)
    raw_results, log_perf = apply_post_select(raw_results, post_select, heralds, True)
    expected = {"results": raw_results,
                "physical_perf": phys_perf,
                "logical_perf": log_perf,
                "global_perf": phys_perf * log_perf,
                KEY_SHOTS_USED: shots_used}

    return expected, sub_results


def test_recombination():
    expected, sub_results = prepare_test()

    averaging = CompilationAveraging(3)

    computation = Computation(CommandFactory.sample_count, Experiment())
    computation.add_params(max_samples = expected["results"].total())

    res = averaging.parse_results(computation, sub_results, NoiseModel())

    assert res == pytest.approx(expected)
