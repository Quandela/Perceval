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

from perceval import SLOSBackend, BasicState, BSDistribution, NoiseModel, PostSelect
from perceval.algorithm import Sampler
from perceval.components import BS, Circuit, FFCircuitProvider, Detector, PERM, Processor, catalog
from perceval.simulators import FFSimulator

backend = SLOSBackend()
sim = FFSimulator(backend)

cnot = FFCircuitProvider(2, 0, Circuit(2)).add_configuration([0, 1], PERM([1, 0]))
detector = Detector.pnr()


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
        circuit_list += [
            ((2 * i,), detector),
            ((2 * i + 1,), detector),
            ((2 * i, 2 * i + 1), cnot)]

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

    proc.add(0, detector)
    proc.add(1, detector)
    proc.add(0, cnot)

    sampler = Sampler(proc)

    assert sampler.probs()["results"] == pytest.approx(BSDistribution(BasicState([0, 1, 0, 1])))


def test_with_herald():
    proc = Processor("SLOS", 5)

    proc.add(0, BS())
    proc.add(1, BS())
    proc.add(1, detector)
    proc.add(2, detector)
    proc.add(1, cnot)

    proc.add_herald(0, 0)

    proc.with_input(BasicState([1, 0, 1, 0]))

    sampler = Sampler(proc)
    res = sampler.probs()
    assert res["results"] == pytest.approx(BSDistribution({
        BasicState([1, 0, 1, 0]): .5,
        BasicState([0, 1, 0, 1]): .5,
    }))

    assert res["global_perf"] == pytest.approx(.5)


def test_with_postselect():
    # Same than with heralds
    proc = Processor("SLOS", 5)

    proc.add(0, BS())
    proc.add(1, BS())
    proc.add(1, detector)
    proc.add(2, detector)
    proc.add(1, cnot)

    proc.with_input(BasicState([0, 1, 0, 1, 0]))
    proc.set_postselection(PostSelect("[0] == 0"))

    sampler = Sampler(proc)
    res = sampler.probs()
    assert res["results"] == pytest.approx(BSDistribution({
        BasicState([0, 1, 0, 1, 0]): .5,
        BasicState([0, 0, 1, 0, 1]): .5,
    }))

    assert res["global_perf"] == pytest.approx(.5)


def test_min_photons_filter():
    config = FFCircuitProvider(2, 0, Circuit(2)).add_configuration([0, 1], BS())

    proc = Processor(backend, 4)

    proc.add(0, BS())
    proc.add(0, detector)
    proc.add(1, detector)
    proc.add(0, config)

    proc.add(3, Detector.threshold())

    proc.min_detected_photons_filter(3)
    input_state = BasicState([1, 0, 1, 1])
    proc.with_input(input_state)  # Equivalent to using a BS with two outputs and a post-selection

    sampler = Sampler(proc)
    expected_perf = .75
    res = sampler.probs()
    assert res["results"] == pytest.approx(BSDistribution({
        input_state: .5 / expected_perf,
        BasicState([0, 1, 2, 0]): .25 / expected_perf
    }))

    assert res["global_perf"] == pytest.approx(expected_perf)


def test_physical_perf():
    # Here the perf is induced by the noise
    proc = Processor("SLOS", 4, noise=NoiseModel(.5))

    proc.add(0, BS())
    proc.add(0, detector)
    proc.add(1, detector)
    proc.add(0, cnot)

    proc.min_detected_photons_filter(2)
    proc.with_input(BasicState([1, 0, 1, 0]))

    sampler = Sampler(proc)
    res = sampler.probs()
    assert res["results"] == pytest.approx(BSDistribution({
        BasicState([1, 0, 1, 0]): 0.5,
        BasicState([0, 1, 0, 1]): 0.5
    }))

    assert res["global_perf"] == pytest.approx(0.25)

    # Here the perf is induced by the circuit and the detectors
    config = FFCircuitProvider(1, 0, BS()).add_configuration([1], Circuit(2))

    proc = Processor("SLOS", 3)

    proc.add(0, BS())
    proc.add(0, Detector.threshold())
    proc.add(0, config)

    proc.add(1, Detector.threshold())
    proc.add(2, Detector.threshold())

    proc.min_detected_photons_filter(2)
    proc.with_input(BasicState([1, 0, 1]))

    sampler = Sampler(proc)
    res = sampler.probs()
    assert res["results"] == pytest.approx(BSDistribution({
        BasicState([1, 0, 1]): 1.,
    }))

    assert res["global_perf"] == pytest.approx(0.5)


def test_with_proc():
    proc = Processor("SLOS", 6)

    cfg = FFCircuitProvider(2, 0, Circuit(4)).add_configuration((0, 1), catalog['postprocessed cnot'].build_processor())
    cnot_perf = 1/9
    proc.add(0, BS())
    proc.add(0, detector)
    proc.add(1, detector)
    proc.add(0, cfg)

    proc.min_detected_photons_filter(2)
    proc.with_input(BasicState([0, 1, 0, 1, 0, 1]))  # ending in logical 11

    sampler = Sampler(proc)
    res = sampler.probs()
    assert res["results"] == pytest.approx(BSDistribution({
        BasicState([1, 0, 0, 1, 0, 1]): .5 / (.5 + .5 * cnot_perf),
        BasicState([0, 1, 0, 1, 1, 0]): .5 * cnot_perf / (.5 + .5 * cnot_perf),
    }))

    assert res["global_perf"] == pytest.approx(.5 * (1 + cnot_perf))

    # Same with heralded processor as default circuit
    proc = Processor("SLOS", 6)

    cfg = FFCircuitProvider(2, 0, catalog['postprocessed cnot'].build_processor())
    cfg.add_configuration((1, 0), Circuit(4))

    cnot_perf = 1 / 9

    proc.add(0, BS())
    proc.add(0, detector)
    proc.add(1, detector)
    proc.add(0, cfg)

    proc.min_detected_photons_filter(2)
    proc.with_input(BasicState([0, 1, 0, 1, 0, 1]))  # ending in logical 11

    sampler = Sampler(proc)
    res = sampler.probs()
    assert res["results"] == pytest.approx(BSDistribution({
        BasicState([1, 0, 0, 1, 0, 1]): .5 / (.5 + .5 * cnot_perf),
        BasicState([0, 1, 0, 1, 1, 0]): .5 * cnot_perf / (.5 + .5 * cnot_perf),
    }))

    assert res["global_perf"] == pytest.approx(.5 * (1 + cnot_perf))


def test_non_adjacent_config():
    proc = Processor("SLOS", 6)

    cnot = FFCircuitProvider(2, 2, Circuit(2)).add_configuration([0, 1], PERM([1, 0]))

    proc.min_detected_photons_filter(2)
    proc.with_input(BasicState([0, 1, 0, 0, 1, 0]))

    proc.add(0, detector)
    proc.add(1, detector)
    proc.add(0, cnot)

    sampler = Sampler(proc)
    assert sampler.probs()["results"] == pytest.approx(BSDistribution(BasicState([0, 1, 0, 0, 0, 1])))

    # Negative offset
    proc = Processor("SLOS", 4)

    cnot = FFCircuitProvider(2, -1, Circuit(2)).add_configuration([0, 1], PERM([1, 0]))

    proc.min_detected_photons_filter(2)
    proc.with_input(BasicState([1, 0, 0, 1]))
    proc.add(2, detector)
    proc.add(3, detector)
    proc.add(2, cnot)

    sampler = Sampler(proc)

    assert sampler.probs()["results"] == pytest.approx(BSDistribution(BasicState([0, 1, 0, 1])))


def test_with_state_vector():
    proc = Processor("SLOS", 4)

    proc.add(0, detector)
    proc.add(1, detector)
    proc.add(0, cnot)

    input_state = (BasicState([1, 0]) + BasicState([0, 1])) * BasicState([1, 0])

    proc.min_detected_photons_filter(2)
    proc.with_input(input_state)
    sampler = Sampler(proc)

    assert sampler.probs()["results"] == pytest.approx(BSDistribution({
        BasicState([1, 0, 1, 0]): .5,
        BasicState([0, 1, 0, 1]): .5
    }))


def test_with_annotated_state_vector():
    proc = Processor("SLOS", 5)

    tri_not = (FFCircuitProvider(2, 0, Circuit(3))
               .add_configuration([0, 2], PERM([2, 1, 0]))
               .add_configuration([1, 1], PERM([1, 2, 0])))

    proc.add(0, BS())
    proc.add(0, detector)
    proc.add(1, detector)
    proc.add(0, tri_not)

    input_state = (BasicState("|{_:0}, {_:1}>") + BasicState("|{_:0}, {_:0}>")) * BasicState("|{_:0}, 0, 0>")

    proc.min_detected_photons_filter(2)
    proc.with_input(input_state)
    sampler = Sampler(proc)

    assert sampler.probs()["results"] == pytest.approx(BSDistribution({
        BasicState([2, 0, 1, 0, 0]): .375,
        BasicState([0, 2, 0, 0, 1]): .375,
        BasicState([1, 1, 0, 1, 0]): .25
    }))


def test_config_with_config():
    proc = Processor("SLOS", 8)

    # Note: please don't do this, this is just to test an edge case
    cnot_proc = Processor("SLOS", 4)
    cnot_proc.add(0, PERM([1, 0]))
    cnot_proc.add(0, Detector.pnr())
    cnot_proc.add(1, Detector.pnr())
    cnot_proc.add(0, cnot)

    double_not = FFCircuitProvider(2, 0, Circuit(2)).add_configuration([0, 1], cnot_proc)

    proc.add(0, BS())
    proc.add(0, Detector.pnr())
    proc.add(1, Detector.pnr())
    proc.add(0, double_not)
    proc.add(4, Detector.pnr())
    proc.add(5, Detector.pnr())
    proc.add(4, cnot)

    proc.min_detected_photons_filter(4)
    proc.with_input(BasicState([1, 0, 1, 0, 1, 0, 1, 0]))

    sampler = Sampler(proc)

    assert sampler.probs()["results"] == pytest.approx(BSDistribution({
        BasicState([1, 0, 1, 0, 1, 0, 1, 0]): .5,
        BasicState([0, 1, 0, 1, 0, 1, 0, 1]): .5
    }))
