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
from perceval.components import Circuit, Processor, BS, Source, catalog, UnavailableModeException, Port, PortLocation
from perceval.utils import BasicState, StateVector, SVDistribution, Encoding
from perceval.backends import Clifford2017Backend


def test_processor_input_generation_0():
    p = Processor("Naive", Circuit(4))  # Init with perfect source
    p.with_input(BasicState([0, 1, 1, 0]))
    assert p.source_distribution == {StateVector([0, 1, 1, 0]): 1}


def test_processor_input_generation_1():
    p = Processor("Naive", Circuit(4), Source(emission_probability=0.2))
    p.with_input(BasicState([0, 1, 1, 0]))
    expected = {
        StateVector([0, 1, 1, 0]): 0.04,
        StateVector([0, 1, 0, 0]): 0.16,
        StateVector([0, 0, 1, 0]): 0.16,
        StateVector([0, 0, 0, 0]): 0.64
    }
    assert pytest.approx(p.source_distribution) == expected


def test_processor_input_generation_2():
    source = Source(emission_probability=0.2,
                    multiphoton_component=0.1, multiphoton_model="indistinguishable",
                    indistinguishability=0.9)
    p = Processor("Naive", Circuit(4), source)
    p.with_input(BasicState([0, 1, 1, 0]))

    expected = {'|0,0,0,0>': 16 / 25,
                '|0,0,2{_:0},0>': 0.0015490319977879558,
                '|0,0,{_:0},0>': 0.15836717690616972,
                '|0,0,{_:0}{_:1},0>': 8.37910960423266e-05,
                '|0,2{_:0},0,0>': 0.0015490319977879558,
                '|0,2{_:0},2{_:0},0>': 3.749218953392102e-06,
                '|0,2{_:0},{_:0},0>': 0.0003636359771584214,
                '|0,2{_:0},{_:0}{_:1},0>': 2.0280482640513694e-07,
                '|0,2{_:0},{_:1},0>': 1.96699985087703e-05,
                '|0,{_:0},0,0>': 0.15836717690616972,
                '|0,{_:0},2{_:0},0>': 0.0003636359771584214,
                '|0,{_:0},2{_:1},0>': 1.96699985087703e-05,
                '|0,{_:0},{_:0},0>': 0.03526897882672976,
                '|0,{_:0},{_:0}{_:1},0>': 1.96699985087703e-05,
                '|0,{_:0},{_:1},0>': 0.0039187754251921985,
                '|0,{_:0},{_:1}{_:2},0>': 1.0640004445062523e-06,
                '|0,{_:0}{_:1},0,0>': 8.37910960423266e-05,
                '|0,{_:0}{_:1},2{_:0},0>': 2.0280482640513694e-07,
                '|0,{_:0}{_:1},{_:0},0>': 1.96699985087703e-05,
                '|0,{_:0}{_:1},{_:0}{_:2},0>': 1.097023089996e-08,
                '|0,{_:0}{_:1},{_:2},0>': 1.0640004445062523e-06}
    result = {str(k): v for k, v in p.source_distribution.items()}
    assert pytest.approx(expected) == result
    assert pytest.approx(sum([v for v in p.source_distribution.values()])) == 1


def test_processor_identity_sv():
    p = Processor("Naive", Circuit(4))  # Init with perfect source
    sv = BasicState([0, 1, 1, 0]) + BasicState([1, 0, 0, 1])
    p.with_input(sv)
    assert p.source_distribution == {sv: 1}


def test_processor_probs():
    source = Source(emission_probability=1, multiphoton_component=0, indistinguishability=1)
    qpu = Processor("Naive", BS(), source)
    qpu.with_input(BasicState([1, 1]))  # Are expected only states with 2 photons in the same mode.
    qpu.thresholded_output(True)  # With thresholded detectors, the simulation will only detect |1,0> and |0,1>
    probs = qpu.probs()
    # By default, all states are filtered and physical performance drops to 0
    assert pytest.approx(probs['physical_perf']) == 0

    qpu.thresholded_output(False)  # With perfect detection, we get our results back
    probs = qpu.probs()
    bsd_out = probs['results']
    assert pytest.approx(bsd_out[BasicState("|2,0>")]) == 0.5
    assert pytest.approx(bsd_out[BasicState("|0,2>")]) == 0.5
    assert pytest.approx(probs['physical_perf']) == 1


def test_processor_samples():
    proc = Processor(Clifford2017Backend(), BS())

    # Without annotations
    proc.with_input(BasicState("|1,1>"))
    samples = proc.samples(500)
    assert samples["results"].count(BasicState([1, 1])) == 0
    assert len(samples["results"]) == 500

    # With annotations
    proc.with_input(SVDistribution({BasicState("|{_:0},{_:1}>"): 1}))
    samples = proc.samples(500)
    assert samples["results"].count(BasicState([1, 1])) > 50


def test_processor_samples_max_shots():
    p = Processor(Clifford2017Backend(), 4)  # Identity circuit with perfect source
    p.with_input(BasicState([1, 1, 1, 1]))
    for (max_samples, max_shots) in [(10, 10), (2, 50), (17, 2), (0, 11), (10, 0)]:
        # In this trivial case, 1 shot = 1 sample, so test this pair of parameters work as a dual threshold system
        samples = p.samples(max_samples, max_shots)
        assert len(samples['results']) == min(max_samples, max_shots)

    p = Processor(Clifford2017Backend(), 4, Source(losses=.92))
    p.add(0, catalog['postprocessed cnot'].build_processor())
    p.with_input(BasicState([0, 1, 0, 1]))
    max_samples = 100
    result_len = {}
    for max_shots in [400, 1_000, 10_000]:
        result_len[max_shots] = len(p.samples(max_samples, max_shots)['results'])
    assert result_len[400] < result_len[1_000]
    assert result_len[1_000] < result_len[10_000]
    assert result_len[10_000] == max_samples  # 10k shots is enough to get the expected sample count


def test_processor_composition():
    p = catalog['postprocessed cnot'].build_processor()  # Circuit with [0,1] and [2,3] post-selection conditions
    p.add((0, 1), BS())  # Composing with a component on modes [0,1] should work
    with pytest.raises(AssertionError):
        p.add((1, 2), BS())  # Composing with a component on modes [1,2] should fail
    p_bs = Processor("SLOS", BS())
    p.add((0, 1), p_bs)  # Composing with a processor on modes [0,1] should work
    with pytest.raises(AssertionError):
        p.add((1, 2), p_bs)  # Composing with a processor on modes [1,2] should fail


def test_add_remove_ports():
    processor = Processor("SLOS", 6)
    p0 = Port(Encoding.DUAL_RAIL, "q0")
    p1 = Port(Encoding.DUAL_RAIL, "q1")
    p2 = Port(Encoding.DUAL_RAIL, "q2")
    processor.add_port(0, p0, PortLocation.OUTPUT)
    processor.add_port(2, p1)
    processor.add_port(4, p2, PortLocation.INPUT)

    with pytest.raises(UnavailableModeException):
        processor.add_port(4, p2, PortLocation.INPUT)

    assert processor.in_port_names == ["", "", "q1", "q1", "q2", "q2"]
    assert processor.out_port_names == ["q0", "q0", "q1", "q1", "", ""]

    assert processor.get_input_port(0) is None
    assert processor.get_input_port(1) is None
    assert processor.get_input_port(2) is p1
    assert processor.get_input_port(3) is p1
    assert processor.get_input_port(4) is p2
    assert processor.get_input_port(5) is p2

    assert processor.get_output_port(0) is p0
    assert processor.get_output_port(1) is p0
    assert processor.get_output_port(2) is p1
    assert processor.get_output_port(3) is p1
    assert processor.get_output_port(4) is None
    assert processor.get_output_port(5) is None

    processor.remove_port(2, PortLocation.OUTPUT)

    with pytest.raises(UnavailableModeException):
        processor.remove_port(2, PortLocation.OUTPUT)

    with pytest.raises(UnavailableModeException):
        processor.add_port(2, p1)

    assert processor.in_port_names == ["", "", "q1", "q1", "q2", "q2"]
    assert processor.out_port_names == ["q0", "q0", "", "", "", ""]

    assert processor.get_input_port(0) is None
    assert processor.get_input_port(1) is None
    assert processor.get_input_port(2) is p1
    assert processor.get_input_port(3) is p1
    assert processor.get_input_port(4) is p2
    assert processor.get_input_port(5) is p2

    assert processor.get_output_port(0) is p0
    assert processor.get_output_port(1) is p0
    assert processor.get_output_port(2) is None
    assert processor.get_output_port(3) is None
    assert processor.get_output_port(4) is None
    assert processor.get_output_port(5) is None

    processor.remove_port(0, PortLocation.OUTPUT)
    processor.remove_port(2, PortLocation.INPUT)
    processor.remove_port(4, PortLocation.INPUT)

    with pytest.raises(UnavailableModeException):
        processor.remove_port(2, PortLocation.INPUT)

    for i in range(6):
        assert processor.get_input_port(i) is None
        assert processor.get_output_port(i) is None
