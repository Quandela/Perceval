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
import time

from flaky import flaky
import pytest
from exqalibur import BSCount, BSSamples

from perceval import SimulatedComputer, Experiment, BS, FockState, BSDistribution, samples_to_sample_count, NoiseModel, \
    ProcessorType, Computation
from perceval.runtime.computation_iterator import ComputationIterator
from tests._test_utils import assert_bsd_close


def test_basic():
    computer = SimulatedComputer("SLOS")

    assert computer.type == ProcessorType.SIMULATOR
    assert computer.noise == NoiseModel()  # Default to no noise
    assert not computer.is_remote
    assert set(computer.available_commands) == {"probs", "samples", "sample_count"}


@flaky(max_runs=3)
def test_probs():
    experiment = Experiment(2)
    experiment.add(0, BS())
    experiment.with_input(FockState([1, 1]))

    computer = SimulatedComputer("SLOS")
    res = computer.probs(experiment)

    expected = BSDistribution({FockState([2, 0]): 0.5, FockState([0, 2]): 0.5})
    assert_bsd_close(res["results"], expected)
    assert res["global_perf"] == pytest.approx(1.)

    computer = SimulatedComputer("CliffordClifford2017")
    res = computer.probs(experiment)

    assert isinstance(res["results"], BSDistribution)
    assert len(res["results"]) == 2
    assert res["results"][FockState([2, 0])] == pytest.approx(0.5, abs = 2.5758 * 0.5 / computer.PROBS_DEFAULT_SAMPLES ** 0.5)
    assert res["results"][FockState([0, 2])] == pytest.approx(0.5, abs = 2.5758 * 0.5 / computer.PROBS_DEFAULT_SAMPLES ** 0.5)


def test_sample_count():
    experiment = Experiment(2)
    experiment.add(0, BS())
    experiment.with_input(FockState([1, 1]))

    n_samples = 10000

    tester = pytest.approx(0.5 * n_samples, abs=3 * 0.5 * n_samples ** 0.5)

    for backend in ["SLOS", "CliffordClifford2017"]:
        computer = SimulatedComputer(backend)
        res = computer.sample_count(experiment, max_samples = n_samples)

        assert isinstance(res["results"], BSCount)
        assert len(res["results"]) == 2
        assert res["results"].total() == n_samples
        assert res["results"][FockState([2, 0])] == tester
        assert res["results"][FockState([0, 2])] == tester
        assert res["global_perf"] == pytest.approx(1.)


def test_samples():
    experiment = Experiment(2)
    experiment.add(0, BS())
    experiment.with_input(FockState([1, 1]))

    n_samples = 10000

    tester = pytest.approx(0.5 * n_samples, abs=3 * 0.5 * n_samples ** 0.5)

    for backend in ["SLOS", "CliffordClifford2017"]:
        computer = SimulatedComputer(backend)
        res = computer.samples(experiment, max_samples=n_samples)

        assert isinstance(res["results"], BSSamples)
        converted = samples_to_sample_count(res["results"])

        assert len(converted) == 2
        assert converted.total() == n_samples
        assert converted[FockState([2, 0])] == tester
        assert converted[FockState([0, 2])] == tester
        assert res["global_perf"] == pytest.approx(1.)


def test_noise():
    experiment = Experiment(2)
    experiment.with_input(FockState([1, 0]))
    experiment.min_detected_photons_filter(1)

    computer = SimulatedComputer("SLOS")
    computer.noise = NoiseModel(0.9)

    res = computer.probs(experiment)
    assert res["results"] == BSDistribution({FockState([1, 0]): 1.})
    assert res["global_perf"] == pytest.approx(0.9)


def test_execute_simple():
    experiment = Experiment(2)
    experiment.with_input(FockState([1, 0]))

    computer = SimulatedComputer("SLOS")
    computation = Computation(computer.get_command("probs"), experiment)
    res = computer.execute(computation)

    assert res["results"] == BSDistribution({FockState([1, 0]): 1.})


def test_execute_async():
    experiment = Experiment(2)
    experiment.with_input(FockState([1, 0]))

    computer = SimulatedComputer("SLOS")
    computation = Computation(computer.get_command("probs"), experiment)

    *access, getters = computer.execute_async(computation)

    assert len(getters) == 1
    assert len(getters[0]) == 1

    while not getters[0][0].is_complete:
        time.sleep(0.01)

    res_out = dict()
    res = computer.get_results(computation, *access, getters, res_out)
    assert res is res_out
    assert res["results"] == BSDistribution({FockState([1, 0]): 1.})

    assert getters[0][0].is_complete


def test_execute_iterator():
    experiment = Experiment(2)
    computer = SimulatedComputer("SLOS")

    computation = Computation(computer.get_command("probs"), experiment)
    computation = ComputationIterator(computation)

    computation.add_iteration(input_state = FockState([1, 0]))
    computation.add_iteration(input_state = FockState([0, 1]))

    res_out = dict()
    res = computer.execute(computation, res_out)

    assert res is res_out

    assert isinstance(res, dict)
    assert "results_list" in res
    assert len(res["results_list"]) == 2

    assert_bsd_close(res["results_list"][0]["results"], BSDistribution(FockState([1, 0])))
    assert_bsd_close(res["results_list"][1]["results"], BSDistribution(FockState([0, 1])))

    assert "iteration" in res["results_list"][0]
    assert res["results_list"][0]["iteration"] == {"input_state": FockState([1, 0])}
    assert res["results_list"][1]["iteration"] == {"input_state": FockState([0, 1])}
