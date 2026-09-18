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
from copy import copy
from unittest.mock import patch

import pytest
from exqalibur import FockState
from exqalibur.exqalibur import PostSelect
from flaky import flaky

import perceval as pcvl
from perceval import (DetectorBalancing, Computation, CommandFactory, Experiment, NoiseModel, apply_min_photons,
                      apply_post_select, Imperfections, Detector, Unitary, tvd_dist)
from perceval.runtime.simulated_computer import SimulatedComputer
from perceval.utils.bsdistribution import BSDistribution
from perceval.utils.constants import KEY_SHOTS_USED
from tests._test_utils import assert_bsd_close


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
    imperfections = Imperfections(NoiseModel(), [])
    averaging = DetectorBalancing()
    comp_list = averaging.extend_computation(computation, imperfections)

    assert len(comp_list) == 1

    assert all(comp.command.name == "probs" for comp in comp_list)


def test_recombination():
    efficiency = 0.1
    expected, sub_results = prepare_test()
    imperfections = Imperfections(NoiseModel(), [Detector.ppnr(wire_efficiency=efficiency)] * 2)

    averaging = DetectorBalancing()

    computation = Computation(CommandFactory.probs, Experiment(2))
    computation.add_params(max_shots = 50000, max_samples = 10000)

    res = averaging.parse_results(computation, sub_results, imperfections)

    assert_bsd_close(res.pop("results"), expected.pop("results"))
    res["global_perf"] *= efficiency ** 2
    res["physical_perf"] *= efficiency ** 2
    assert pytest.approx(res) == expected


@flaky(max_runs=3)
def test_run_through():
    experiment = Experiment(Unitary.random(4))
    experiment.with_input(FockState([1, 0, 1, 0]))
    experiment.min_detected_photons_filter(1)

    c = SimulatedComputer("SLOS")
    c.compute_physical_logical_perf(True)
    c.noise = NoiseModel(0.05, 0.8, 0.03)
    computation = Computation(c.get_command("probs"), experiment)

    perfect_res = c.execute(computation)

    for m in range(experiment.circuit_size):
        experiment.add(m, Detector.ppnr(n_wires = 24, wire_efficiency = 0.5 + random.random() / 2))

    noisy_res = c.execute(computation)
    with pytest.raises(AssertionError):
        assert_bsd_close(perfect_res["results"], noisy_res["results"])

    c.mitigations = [DetectorBalancing()]
    mitigated_res = c.execute(computation)
    assert_bsd_close(perfect_res["results"], mitigated_res["results"], rel=1e-4)
    assert pytest.approx(perfect_res["global_perf"], rel=1e-4) == mitigated_res["global_perf"]
    assert pytest.approx(perfect_res["physical_perf"], rel=1e-4) == mitigated_res["physical_perf"]


@patch.object(pcvl.utils.logging.ExqaliburLogger, "warn")  # Suppress the warning in tvd_dist()
def test_run_through_with_max_detection(mock_warn):
    # With a max detection value, we can't completely suppress the detector noise, but we still lower the tvd
    experiment = Experiment(Unitary.random(4))
    experiment.with_input(FockState([1, 0, 1, 0]))
    experiment.min_detected_photons_filter(1)

    c = SimulatedComputer("SLOS")
    c.noise = NoiseModel(0.05, 0.8, 0.03)
    computation = Computation(c.get_command("probs"), experiment)

    perfect_res = c.execute(computation)["results"]

    for m in range(experiment.circuit_size):
        experiment.add(m, Detector.ppnr(n_wires = 24,
                                        max_detections = 2,
                                        wire_efficiency = 0.5 + random.random() / 2))

    noisy_res = c.execute(computation)["results"]
    tvd_noisy = tvd_dist(perfect_res, noisy_res)

    c.mitigations = [DetectorBalancing()]
    mitigated_res = c.execute(computation)["results"]
    tvd_mitigated = tvd_dist(perfect_res, mitigated_res)

    assert tvd_mitigated < tvd_noisy
