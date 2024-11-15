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

from perceval import SLOSBackend, Circuit, PERM, BasicState, BS, BSDistribution, Processor, Detector, NoiseModel
from perceval.algorithm import Sampler
from perceval.components.feed_forward_configurator import CircuitMapFFConfig
from perceval.simulators.feed_forward_simulator import FFSimulator

backend = SLOSBackend()
sim = FFSimulator(backend)

cnot = CircuitMapFFConfig(2, 0, Circuit(2)).add_configuration([0, 1], PERM([1, 0]))


def test_basic_circuit():

    sim.set_circuit([((0, 1), cnot)])

    assert sim.probs(BasicState([1, 0, 1, 0]))[BasicState([1, 0, 1, 0])] == pytest.approx(1.)
    assert sim.probs(BasicState([1, 0, 0, 1]))[BasicState([1, 0, 0, 1])] == pytest.approx(1.)
    assert sim.probs(BasicState([0, 1, 1, 0]))[BasicState([0, 1, 0, 1])] == pytest.approx(1.)
    assert sim.probs(BasicState([0, 1, 0, 1]))[BasicState([0, 1, 1, 0])] == pytest.approx(1.)


def test_cascade():
    n = 3

    circuit_list = [((0, 1), BS())]
    for i in range(n):
        circuit_list += [((2 * i, 2 * i + 1), cnot)]

    sim.set_circuit(circuit_list)
    input_state = BasicState((n + 1) * [1, 0])

    assert sim.probs(input_state) == pytest.approx(BSDistribution({
        input_state: .5,
        BasicState((n + 1) * [0, 1]): .5
    })), "Incorrect simulated distribution"


def test_with_processor():
    # Same than test_basic_circuit, but using a Processor and a sampler to get the results
    proc = Processor("SLOS", 4)

    proc.min_detected_photons_filter(2)
    proc.with_input(BasicState([0, 1, 1, 0]))

    proc.add(0, cnot)

    sampler = Sampler(proc)

    assert sampler.probs()["results"] == pytest.approx(BSDistribution(BasicState([0, 1, 0, 1])))


def test_with_herald():
    proc = Processor("SLOS", 5)

    proc.add(0, BS())
    proc.add(1, BS())
    proc.add(1, cnot)

    proc.add_herald(0, 0)

    proc.min_detected_photons_filter(2)
    proc.with_input(BasicState([1, 0, 1, 0]))

    sampler = Sampler(proc)
    assert sampler.probs()["results"] == pytest.approx(BSDistribution({
        BasicState([1, 0, 1, 0]): .5,
        BasicState([0, 1, 0, 1]): .5,
    }))


def test_min_photons_filter():
    config = CircuitMapFFConfig(2, 0, Circuit(2)).add_configuration([0, 1], BS())

    proc = Processor(backend, 5)

    proc.add(0, BS())
    proc.add(1, BS())
    proc.add(1, config)

    proc.add_herald(0, 0)
    proc.add(4, Detector.threshold())

    proc.min_detected_photons_filter(1)
    proc.with_input(BasicState([1, 0, 1, 2]))  # Equivalent to using a BS with two outputs and a post-selection

    sampler = Sampler(proc)
    assert len(sampler.probs()["results"]) == 5


def test_physical_perf():
    proc = Processor("SLOS", 4, noise=NoiseModel(.5))

    proc.add(0, BS())
    proc.add(0, cnot)

    proc.min_detected_photons_filter(2)
    proc.with_input(BasicState([1, 0, 1, 0]))

    sampler = Sampler(proc)
    res = sampler.probs()
    assert res["results"] == pytest.approx(BSDistribution({
        BasicState([1, 0, 1, 0]): 0.5,
        BasicState([0, 1, 0, 1]): 0.5
    }))

    assert res["physical_perf"] == pytest.approx(0.25)
    assert res["logical_perf"] == pytest.approx(1.)
