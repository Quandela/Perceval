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

from copy import copy
import random

import pytest
from exqalibur import FockState
from exqalibur.exqalibur import PostSelect

from perceval import DetectorBalancing, Computation, CommandFactory, Experiment, NoiseModel, Command, BSCount, \
    apply_min_photons, apply_post_select
from perceval.runtime.simulated_computer import SimulatedComputer
from perceval.utils.bsdistribution import BSDistribution
from perceval.utils.constants import KEY_SHOTS_USED
from perceval.utils.states import BasicState

def prepare_test():
    raw_results = BSDistribution({FockState([0, 1]): 1,
                                  FockState([1, 0]): 1,
                                  FockState([1, 1]): 1,
                                  FockState([2, 0]): 1,
                                  FockState([0, 2]): 1})
    raw_results.normalize()

    shots_used = 1_000_000
    min_photons = 2
    post_select = PostSelect("[1] >= 1")
    heralds = {}

    raw_results, phys_perf = apply_min_photons(raw_results, min_photons)
    raw_results, log_perf = apply_post_select(raw_results, post_select, heralds, True)

    sub_results = []

    sub_results.append({"results": copy(raw_results),
                        "physical_perf": phys_perf,
                        "logical_perf": log_perf,
                        "global_perf": phys_perf * log_perf,
                        KEY_SHOTS_USED: shots_used})

    expected = {"results": raw_results,
                "physical_perf": phys_perf,
                "logical_perf": log_perf,
                "global_perf": phys_perf * log_perf,
                KEY_SHOTS_USED: shots_used}

    return expected, sub_results


def test_computation_extension():
    computation = Computation(CommandFactory.samples, Experiment())
    averaging = DetectorBalancing()
    comp_list = averaging.extend_computation(computation, NoiseModel())

    assert len(comp_list) == 1

    assert all(comp.command.name == "probs" for comp in comp_list)


def test_recombination():
    expected, sub_results = prepare_test()
    noise = NoiseModel()
    noise.transmitance_ratios_output = [1., 1.]

    averaging = DetectorBalancing()

    computation = Computation(CommandFactory.probs, Experiment(2))
    computation.add_params(max_shots = 50000, max_samples = 10000)

    res = averaging.parse_results(computation, sub_results, noise)

    assert res == expected


def test_run_through():
    computer = SimulatedComputer("SLOS")
    computer.mitigations = [ DetectorBalancing() ]
    noise = NoiseModel()
    noise.transmitance_ratios_output = [1., .5]
    computer.noise = noise

    e = Experiment(2)
    e.min_detected_photons_filter(1)
    e.with_input(BasicState([1, 0]))
    computation = Computation(CommandFactory.probs, e)
    computation.add_params(max_shots = 50000, max_samples = 10000)

    computer.execute(computation)
