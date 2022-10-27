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


def test_processor_generator_0():
    p = pcvl.Processor("Naive", pcvl.Circuit(4))  # Init with perfect source
    p.with_input(pcvl.BasicState([0, 1, 1, 0]))
    assert p.source_distribution == {pcvl.StateVector([0, 1, 1, 0]): 1}


def test_processor_generator_1():
    p = pcvl.Processor("Naive", pcvl.Circuit(4), pcvl.Source(brightness=0.2))
    p.with_input(pcvl.BasicState([0, 1, 1, 0]))
    expected = {
                pcvl.StateVector([0, 1, 1, 0]): 0.04,
                pcvl.StateVector([0, 1, 0, 0]): 0.16,
                pcvl.StateVector([0, 0, 1, 0]): 0.16,
                pcvl.StateVector([0, 0, 0, 0]): 0.64
               }
    assert pytest.approx(p.source_distribution) == expected


def test_processor_generator_2():
    source = pcvl.Source(brightness=0.2,
                         purity=0.9, purity_model="indistinguishable",
                         indistinguishability=0.9, indistinguishability_model="linear")
    p = pcvl.Processor("Naive", pcvl.Circuit(4), source)
    p.with_input(pcvl.BasicState([0, 1, 1, 0]))
    expected = {
        "|0,0,0,0>": 16/25,
        "|0,0,2{_:0},0>":  0.016,
        "|0,0,{_:2},0>":  0.0144,
        "|0,0,{_:0},0>": 0.1296,
        "|0,2{_:0},0,0>": 0.016,
        "|0,2{_:0},2{_:0},0>": 4e-4,
        "|0,2{_:0},{_:2},0>": 3.6e-4,
        "|0,2{_:0},{_:0},0>": 0.00324,
        "|0,{_:1},0,0>": 0.0144,
        "|0,{_:1},2{_:0},0>": 3.6e-4,
        "|0,{_:1},{_:2},0>": 3.24e-4,
        "|0,{_:1},{_:0},0>": 0.002916,
        "|0,{_:0},0,0>": 0.1296,
        "|0,{_:0},2{_:0},0>": 0.00324,
        "|0,{_:0},{_:2},0>": 0.002916,
        "|0,{_:0},{_:0},0>": 0.026244
    }
    result = {str(k): v for k, v in p.source_distribution.items()}
    assert pytest.approx(expected) == result
    assert pytest.approx(sum([v for v in p.source_distribution.values()])) == 1


def test_processor_probs():
    source = pcvl.Source(brightness=1, purity=1, indistinguishability=1)
    qpu = pcvl.Processor("Naive", comp.BS(), source)
    qpu.with_input(pcvl.BasicState([1, 1]))  # Are expected only states with 2 photons in the same mode.
    probs = qpu.probs()
    # By default, all states are filtered and physical performance drops to 0
    assert pytest.approx(probs['physical_perf']) == 0

    qpu.mode_post_selection(1)  # Lower mode_post_selection to 1 (default was 2 = expected input photon count)
    probs = qpu.probs()
    bsd_out = probs['results']
    assert pytest.approx(bsd_out[pcvl.BasicState("|2,0>")]) == 0.5
    assert pytest.approx(bsd_out[pcvl.BasicState("|0,2>")]) == 0.5
