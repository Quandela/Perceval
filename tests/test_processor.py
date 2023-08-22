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
import perceval as pcvl
import perceval.components.unitary_components as comp
from perceval.backends import Clifford2017Backend


def test_processor_generator_0():
    p = pcvl.Processor("Naive", pcvl.Circuit(4))  # Init with perfect source
    p.with_input(pcvl.BasicState([0, 1, 1, 0]))
    assert p.source_distribution == {pcvl.StateVector([0, 1, 1, 0]): 1}


def test_processor_generator_1():
    p = pcvl.Processor("Naive", pcvl.Circuit(4), pcvl.Source(emission_probability=0.2))
    p.with_input(pcvl.BasicState([0, 1, 1, 0]))
    expected = {
                pcvl.StateVector([0, 1, 1, 0]): 0.04,
                pcvl.StateVector([0, 1, 0, 0]): 0.16,
                pcvl.StateVector([0, 0, 1, 0]): 0.16,
                pcvl.StateVector([0, 0, 0, 0]): 0.64
               }
    assert pytest.approx(p.source_distribution) == expected


def test_processor_generator_2():
    source = pcvl.Source(emission_probability=0.2,
                         multiphoton_component=0.1, multiphoton_model="indistinguishable",
                         indistinguishability=0.9)
    p = pcvl.Processor("Naive", pcvl.Circuit(4), source)
    p.with_input(pcvl.BasicState([0, 1, 1, 0]))

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
    p = pcvl.Processor("Naive", pcvl.Circuit(4))  # Init with perfect source
    sv = pcvl.BasicState([0, 1, 1, 0]) + pcvl.BasicState([1, 0, 0, 1])
    p.with_input(sv)
    assert p.source_distribution == {sv: 1}


def test_processor_probs():
    source = pcvl.Source(emission_probability=1, multiphoton_component=0, indistinguishability=1)
    qpu = pcvl.Processor("Naive", comp.BS(), source)
    qpu.with_input(pcvl.BasicState([1, 1]))  # Are expected only states with 2 photons in the same mode.
    qpu.thresholded_output(True)  # With thresholded detectors, the simulation will only detect |1,0> and |0,1>
    probs = qpu.probs()
    # By default, all states are filtered and physical performance drops to 0
    assert pytest.approx(probs['physical_perf']) == 0

    qpu.thresholded_output(False) # With perfect detection, we get our results back
    probs = qpu.probs()
    bsd_out = probs['results']
    assert pytest.approx(bsd_out[pcvl.BasicState("|2,0>")]) == 0.5
    assert pytest.approx(bsd_out[pcvl.BasicState("|0,2>")]) == 0.5
    assert pytest.approx(probs['physical_perf']) == 1


def test_processor_samples():
    proc = pcvl.Processor(Clifford2017Backend(), comp.BS())

    # Without annotations
    proc.with_input(pcvl.BasicState("|1,1>"))
    samples = proc.samples(500)
    assert samples["results"].count(pcvl.BasicState([1, 1])) == 0

    # With annotations
    proc.with_input(pcvl.SVDistribution({pcvl.BasicState("|{_:0},{_:1}>"): 1}))
    samples = proc.samples(500)
    assert samples["results"].count(pcvl.BasicState([1,1])) > 50
