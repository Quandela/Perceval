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

from perceval import NoiseModel
from perceval.algorithm.sampler import Sampler
import perceval as pcvl
from perceval.components import BS, PS, Processor, catalog


# To speed up the tests, lower the sample count required to compute a probability distribution
Sampler.PROBS_SIMU_SAMPLE_COUNT = 1000


@pytest.mark.parametrize("backend_name", ["SLOS", "CliffordClifford2017"])  # MPS cannot be used with >2-modes components
def test_sampler_standard(backend_name):
    TRANSMITTANCE = 0.9
    noise_model = NoiseModel(brightness=TRANSMITTANCE)
    p = catalog['postprocessed cnot'].build_processor(backend=backend_name)
    p.noise = noise_model
    p.min_detected_photons_filter(0)
    p.with_input(pcvl.BasicState([1, 0, 1, 0]))
    sampler = Sampler(p)
    probs = sampler.probs()
    assert probs['results'][pcvl.BasicState([1, 0, 1, 0])] == pytest.approx(1)
    assert probs['results'][pcvl.BasicState([1, 0, 0, 1])] == pytest.approx(0)
    samples = sampler.samples(4)
    assert len(samples['results']) == 4
    sample_count = sampler.sample_count(100)
    assert 90 < sum(list(sample_count['results'].values())) < 110


def test_sampler_missing_input_state():
    p = Processor("SLOS", BS())
    sampler = Sampler(p)
    with pytest.raises(AssertionError):
        sampler.probs.execute_async()
    with pytest.raises(AssertionError):
        sampler.sample_count.execute_async(100)
    with pytest.raises(AssertionError):
        sampler.samples.execute_async(100)

    p.with_input(pcvl.BasicState([1, 1]))  # When setting a valid input state, the simulation works
    assert isinstance(sampler.probs()['results'], pcvl.BSDistribution)


def test_sampler_iteration_missing_input_state():
    sampler = Sampler(Processor("SLOS", BS(pcvl.P("theta0"))))
    sampler.add_iteration_list([{"circuit_params": {"theta0": x}} for x in range(0, 4)])
    with pytest.raises(AssertionError):
        sampler.probs.execute_async()

    sampler = Sampler(Processor("SLOS", BS(pcvl.P("theta0"))))
    sampler.add_iteration_list([{"circuit_params": {"theta0": 0}, "input_state": pcvl.BasicState([1, 1])},
                                {"circuit_params": {"theta0": 1}, "input_state": pcvl.BasicState([1, 1])},
                                {"circuit_params": {"theta0": 1}}  # input state is missing
                                ])
    with pytest.raises(AssertionError):
        sampler.probs.execute_async()


def test_sampler_iteration_bad_params():
    c = BS() // PS(phi=pcvl.P("phi1")) // BS()  # circuit with only one variable parameter: phi1
    p = Processor("SLOS", c)
    sampler = Sampler(p)

    with pytest.raises(AssertionError):
        sampler.add_iteration(circuit_params="phi0")  # wrong parameter type
    with pytest.raises(AssertionError):
        sampler.add_iteration(circuit_params={"phi0": 1})  # phi0 doesn't exist in the input circuit
    with pytest.raises(AssertionError):
        sampler.add_iteration(circuit_params={"phi0": 1, "phi1": 2})  # phi0 doesn't exist in the input circuit
    with pytest.raises(AssertionError):
        sampler.add_iteration(circuit_params={"phi1": "phi"})  # phi is not an acceptable value
    with pytest.raises(AssertionError):
        sampler.add_iteration(input_state=pcvl.BasicState([1]))  # input state too short
    with pytest.raises(AssertionError):
        sampler.add_iteration(input_state=pcvl.BasicState([1, 0, 0]))  # input state too large


def test_sampler_clear_iterations():
    c = BS() // PS(phi=pcvl.P("phi1")) // BS()
    p = pcvl.Processor("SLOS", c)
    sampler = Sampler(p)
    iteration_list = [
        {"circuit_params": {"phi1": 0.5}, "input_state": pcvl.BasicState([1, 1]), "min_detected_photons": 1},
        {"circuit_params": {"phi1": 0.9}, "input_state": pcvl.BasicState([1, 1]), "min_detected_photons": 1},
        {"circuit_params": {"phi1": 1.57}, "input_state": pcvl.BasicState([1, 0]), "min_detected_photons": 1}
    ]
    assert sampler.n_iterations == 0

    sampler.add_iteration_list(iteration_list)
    assert sampler.n_iterations == len(iteration_list)

    sampler.add_iteration_list(iteration_list)
    assert sampler.n_iterations == len(iteration_list)*2

    sampler.clear_iterations()
    assert sampler.n_iterations == 0

    sampler.add_iteration(circuit_params={"phi1": 0.1}, input_state=pcvl.BasicState([0, 1]))
    assert sampler.n_iterations == 1


@pytest.mark.parametrize("backend_name", ["SLOS", "CliffordClifford2017"])
def test_sampler_iterator(backend_name):
    phi = pcvl.P("phi1")
    c = BS() // PS(phi=phi) // BS()
    phi.set_value(0)
    p = pcvl.Processor(backend_name, c, noise=pcvl.NoiseModel(.5))
    p.min_detected_photons_filter(2)
    p.with_input(pcvl.BasicState([1, 1]))
    sampler = Sampler(p)
    iteration_list = [
        {"circuit_params": {"phi1": 0.5}, "input_state": pcvl.BasicState([1, 1]), "min_detected_photons": 1},
        {"circuit_params": {"phi1": 0.9}, "input_state": pcvl.BasicState([1, 1]), "max_samples": 20},
        {"circuit_params": {"phi1": 1.57}, "input_state": pcvl.BasicState([1, 0]), "min_detected_photons": 1, "max_shots": 30},
        {"circuit_params": {"phi1": 1.57}, "input_state": pcvl.BasicState([1, 0]), "min_detected_photons": 1, "noise": pcvl.NoiseModel()},
        {}  # Test default parameters
    ]
    sampler.add_iteration_list(iteration_list)
    rl = sampler.probs()['results_list']
    assert len(rl) == len(iteration_list)
    for i in range(len(iteration_list)):
        assert "results" in rl[i]
        assert "iteration" in rl[i]
        assert rl[i]["iteration"] == iteration_list[i]
    # Test that the results changes given the iteration parameters (avoid Clifford as sampling adds randomness)
    if backend_name == "SLOS":
        assert len(rl[0]["results"]) == 5  # |0, 1>, |1, 0>, |2, 0>, |1, 1> and |0, 2>
        assert rl[1]["results"][pcvl.BasicState([1, 1])] == pytest.approx(0.38639895265345636)
        assert len(rl[2]["results"]) == 2  # |0, 1> and |1, 0>
        assert rl[2]["physical_perf"] == pytest.approx(.5)
        assert rl[4]["physical_perf"] == pytest.approx(.25)
    assert rl[3]["physical_perf"] == pytest.approx(1.)
    assert rl[4]["results"][pcvl.BasicState([1, 1])] == pytest.approx(1.)

    res = sampler.samples(max_samples=10)
    assert len(res['results_list']) == len(iteration_list)
    assert len(res['results_list'][0]["results"]) == 10
    assert len(res['results_list'][1]["results"]) == 20
    assert len(res['results_list'][2]["results"]) == 10
    assert rl[3]["physical_perf"] == pytest.approx(1.)
    assert len(res['results_list'][4]["results"]) == 10

    res = sampler.sample_count(max_samples=100)
    assert len(res['results_list']) == len(iteration_list)
    assert sum(res['results_list'][0]["results"].values()) == 100
    assert sum(res['results_list'][1]["results"].values()) == 20
    assert sum(res['results_list'][2]["results"].values()) == 30
    assert rl[3]["physical_perf"] == pytest.approx(1.)
    assert sum(res['results_list'][4]["results"].values()) == 100

    # Test wrong parameters
    if backend_name == "SLOS":  # No need to do it twice
        n_it = sampler.n_iterations
        with pytest.raises(NotImplementedError):
            sampler.add_iteration(not_implemented_param = 0)
        with pytest.raises(AssertionError):
            sampler.add_iteration(input_state=[1, 0])  # Wrong type
        with pytest.raises(AssertionError):
            sampler.add_iteration(input_state=pcvl.BasicState([1]))  # Wrong number of modes
        with pytest.raises(AssertionError):
            sampler.add_iteration(circuit_params={"phi0" : 1})  # Non-existing parameter
        with pytest.raises(AssertionError):
            sampler.add_iteration(circuit_params={"phi1" : "phi"})  # Not a number
        assert sampler.n_iterations == n_it  # No new iteration


def test_iterator_with_heralds():
    c = pcvl.catalog['postprocessed cnot'].build_processor()

    processor = pcvl.Processor("SLOS")
    processor.add(0, c)

    processor.with_input(pcvl.BasicState([0, 1, 0, 1]))
    processor.min_detected_photons_filter(1)

    sampler = Sampler(processor, max_shots_per_call=500)
    sampler.add_iteration(input_state=pcvl.BasicState([1, 0, 1, 0]))
    sampler.add_iteration(input_state=pcvl.BasicState([0, 1, 0, 1]))
    sampler.add_iteration(input_state=pcvl.BasicState([0, 1, 1, 0]))
    sampler.add_iteration(input_state=pcvl.BasicState([1, 0, 0, 1]))

    res = sampler.probs()['results_list']

    assert res[0]["results"][pcvl.BasicState([1, 0, 1, 0])] == pytest.approx(1.)
    assert res[1]["results"][pcvl.BasicState([0, 1, 1, 0])] == pytest.approx(1.)
    assert res[2]["results"][pcvl.BasicState([0, 1, 0, 1])] == pytest.approx(1.)
    assert res[3]["results"][pcvl.BasicState([1, 0, 0, 1])] == pytest.approx(1.)


@pytest.mark.parametrize("backend_name", ["SLOS", "Naive", "MPS", "CliffordClifford2017"])
def test_sampler_shots(backend_name):
    p = Processor(backend_name, BS(theta=0.8))
    p.with_input(pcvl.BasicState([1, 1]))
    p.thresholded_output(True)
    sampler = Sampler(p)  # Without a max_shots_per_call value
    samples = sampler.samples(max_samples=100)
    assert len(samples['results']) == 100  # You get the number of samples you asked for

    sampler = Sampler(p, max_shots_per_call=10)
    samples = sampler.samples(max_samples=100)
    assert len(samples['results']) <= 10
    samples = sampler.samples()
    assert len(samples['results']) <= 10
