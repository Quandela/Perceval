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

import numpy as np
import pytest
import perceval as pcvl
import perceval.lib.symb as symb


# Utils
def outputstate_to_2outcome(output):
    """
    :param output: an output of the chip
    :return: a measurable outcome
    """
    state = []
    for m in output:
        if m.isdigit():
            state.append(m)

    if int(state[0]) == 0 and int(state[1]) == 0:
        return '|0,0>'
    if int(state[0]) == 0 and int(state[1]) > 0:
        return '|0,1>'
    if int(state[0]) > 0 and int(state[1]) == 0:
        return '|1,0>'
    if int(state[0]) > 0 and int(state[1]) > 0:
        return '|1,1>'


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
    mzi_chip = pcvl.Circuit(m=2, name="MZI")
    phase_shifters = [pcvl.Parameter("phi1"), pcvl.Parameter("phi2")]
    (mzi_chip
     .add(0, symb.PS(phase_shifters[0]))
     .add((0, 1), symb.BS())
     .add(0, symb.PS(phase_shifters[1]))
     .add((0, 1), symb.BS())
     )
    # Initial phase set to zero
    phase_shifters[0].set_value(0)
    # Internal phase set to pi/2
    phase_shifters[1].set_value(np.pi / 2)
    source = pcvl.Source(brightness=1,
                         multiphoton_component=0.01,
                         multiphoton_model="distinguishable",
                         indistinguishability=0.9,
                         indistinguishability_model="homv")
    p = pcvl.Processor({0: source, 1: source}, mzi_chip)

    expected = {'|{_:0}{_:1},{_:0}{_:1}>': 9.183204944499092e-05,
                '|{_:0}{_:1},{_:1}{_:3}>': 4.967430037466685e-06,
                '|{_:0}{_:1},{_:3}>': 0.0004892793143552383,
                '|{_:0}{_:1},{_:0}>': 0.009045224965301355,
                '|{_:1}{_:2},{_:0}{_:1}>': 4.967430037466685e-06,
                '|{_:1}{_:2},{_:1}{_:3}>': 2.6870097451007303e-07,
                '|{_:1}{_:2},{_:3}>': 2.646636743411902e-05,
                '|{_:1}{_:2},{_:0}>': 0.0004892793143552384,
                '|{_:2},{_:0}{_:1}>': 0.0004892793143552383,
                '|{_:2},{_:1}{_:3}>': 2.646636743411902e-05,
                '|{_:2},{_:3}>': 0.0026068703562946577,
                '|{_:2},{_:0}>': 0.04819277687864922,
                '|{_:0},{_:0}{_:1}>': 0.009045224965301355,
                '|{_:0},{_:1}{_:3}>': 0.0004892793143552384,
                '|{_:0},{_:3}>': 0.04819277687864922,
                '|{_:0},{_:0}>': 0.8909318170223374}

    result = {str(k): v for k, v in p.source_distribution.items()}
    assert pytest.approx(expected) == result


def test_processor_run_1():
    simulator_backend = pcvl.BackendFactory().get_backend('Naive')
    source = pcvl.Source(brightness=1, multiphoton_component=0, indistinguishability=1)
    qpu = pcvl.Processor({0: source, 1: source}, symb.BS())
    all_p, sv_out = qpu.run(simulator_backend)
    assert pytest.approx(all_p) == 1
    assert pytest.approx(sv_out[pcvl.StateVector("|2,0>")]) == 0.5
    assert pytest.approx(sv_out[pcvl.StateVector("|0,2>")]) == 0.5


def test_processor_run_2():
    simulator_backend = pcvl.BackendFactory().get_backend('Naive')
    mzi_chip = pcvl.Circuit(m=2, name="MZI")
    phase_shifters = [pcvl.Parameter("phi1"), pcvl.Parameter("phi2")]
    (mzi_chip
     .add(0, symb.PS(phase_shifters[0]))
     .add((0, 1), symb.BS())
     .add(0, symb.PS(phase_shifters[1]))
     .add((0, 1), symb.BS())
     )
    # Initial phase set to zero
    phase_shifters[0].set_value(0)
    # Internal phase set to pi/2
    phase_shifters[1].set_value(np.pi / 2)

    for g2 in np.linspace(0.0, 0.3, 5):
        for M in np.linspace(0.8, 1, 5):
            outcome = {'|0,0>': 0,
                       '|1,0>': 0,
                       '|1,1>': 0,
                       '|0,1>': 0
                       }
            source = pcvl.Source(brightness=1,
                                 multiphoton_component=g2,
                                 multiphoton_model="distinguishable",
                                 indistinguishability=M,
                                 indistinguishability_model="homv")
            p = pcvl.Processor({0: source, 1: source, }, mzi_chip)

            all_p, sv_out = p.run(simulator_backend)

            for output_state in sv_out:
                # Each output is mapped to an outcome
                result = outputstate_to_2outcome(str(output_state))
                # The probability of an outcome is added, weighted by the probability of this input
                outcome[result] += sv_out[output_state]

            visibility = 1 - 2 * outcome['|1,1>'] / (
                        outcome['|0,0>'] + outcome['|1,0>'] + outcome['|1,1>'] + outcome['|0,1>'])

            assert pytest.approx(visibility) == M * (1 - g2) - g2
