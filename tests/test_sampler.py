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

from perceval.algorithm.sampler import Sampler
import perceval as pcvl
from perceval.components import BS, PS, Processor, Source

import numpy as np


def test_sampler():
    theta_r13 = BS.r_to_theta(1 / 3)
    cnot = pcvl.Circuit(6, name="Ralph CNOT")
    cnot.add((0, 1), BS.H(theta=theta_r13, phi_bl=np.pi, phi_tr=np.pi / 2, phi_tl=-np.pi / 2))
    cnot.add((3, 4), BS.H())
    cnot.add((2, 3), BS.H(theta=theta_r13, phi_bl=np.pi, phi_tr=np.pi / 2, phi_tl=-np.pi / 2))
    cnot.add((4, 5), BS.H(theta=theta_r13))
    cnot.add((3, 4), BS.H())
    imperfect_source = Source(emission_probability=0.9)

    for backend_name in ['CliffordClifford2017', 'Naive', 'SLOS', 'MPS']:
        p = Processor(backend_name, cnot, imperfect_source)
        p.min_detected_photons_filter(1)
        p.with_input(pcvl.BasicState([1, 0, 1, 0, 1, 0]))
        sampler = Sampler(p)
        probs = sampler.probs()
        assert probs['results'][pcvl.BasicState('|0,1,2,0,0,0>')] > 0.1
        assert probs['results'][pcvl.BasicState('|0,1,0,0,2,0>')] > 0.1
        samples = sampler.samples(50)
        assert len(samples['results']) == 50
        sample_count = sampler.sample_count(1000)
        assert 940 < sum(list(sample_count['results'].values())) < 1060


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
        sampler.add_iteration()  # Parameters are required for an iteration
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

def test_sampler_iterator():
    c = BS() // PS(phi=pcvl.P("phi1")) // BS()
    for backend_name in ["SLOS", "CliffordClifford2017"]:
        p = pcvl.Processor(backend_name, c)
        sampler = Sampler(p)
        iteration_list = [
            {"circuit_params": {"phi1": 0.5}, "input_state": pcvl.BasicState([1, 1]), "min_detected_photons": 1},
            {"circuit_params": {"phi1": 0.9}, "input_state": pcvl.BasicState([1, 1]), "min_detected_photons": 1},
            {"circuit_params": {"phi1": 1.57}, "input_state": pcvl.BasicState([1, 0]), "min_detected_photons": 1}
        ]
        sampler.add_iteration_list(iteration_list)
        rl = sampler.probs()['results_list']
        assert len(rl) == len(iteration_list)
        # Test that the results changes given the iteration parameters (avoid Clifford as sampling adds randomness)
        if backend_name == "SLOS":
            assert rl[0]["results"][pcvl.BasicState([1, 1])] == pytest.approx(0.7701511529340699)
            assert rl[1]["results"][pcvl.BasicState([1, 1])] == pytest.approx(0.38639895265345636)
            assert rl[2]["results"][pcvl.BasicState([1, 1])] == pytest.approx(0)
        res = sampler.samples(10)
        assert len(res['results_list']) == len(iteration_list)
        res = sampler.sample_count(100)
        assert len(res['results_list']) == len(iteration_list)
