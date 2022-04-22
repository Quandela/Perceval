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
import perceval.lib.symb as symb


def test_processor_generator_0():
    p = pcvl.Processor({1: pcvl.Source(), 2: pcvl.Source()}, pcvl.Circuit(4))
    assert p.source_distribution == {pcvl.StateVector([0, 1, 1, 0]): 1}


def test_processor_generator_1():
    p = pcvl.Processor({1: pcvl.Source(brightness=0.2), 2: pcvl.Source(brightness=0.2)}, pcvl.Circuit(4))
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
    p = pcvl.Processor({1: source, 2: source}, pcvl.Circuit(4))
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


def test_processor_run():
    simulator_backend = pcvl.BackendFactory().get_backend('Naive')
    source = pcvl.Source(brightness=1, purity=1, indistinguishability=1)
    qpu = pcvl.Processor({0: source, 1: source}, symb.BS())
    all_p, sv_out = qpu.run(simulator_backend)
    assert pytest.approx(all_p) == 1
    assert pytest.approx(sv_out[pcvl.StateVector("|2,0>")]) == 0.5
    assert pytest.approx(sv_out[pcvl.StateVector("|0,2>")]) == 0.5
