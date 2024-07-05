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

from perceval.simulators.delay_simulator import _retrieve_mode_count, DelaySimulator
from perceval.simulators.simulator import Simulator
from perceval.backends._naive import NaiveBackend
from perceval.components import Circuit, BS, TD
from perceval.utils import BasicState, BSDistribution, PostSelect

import pytest


def test_retrieve_mode_count():
    comp_list = [((0,1), None), ((1,2,3), None), ((0,1), None), ((0,1), None), ((0,), None)]
    assert _retrieve_mode_count(comp_list) == 4


def test_prepare_circuit():
    sim = DelaySimulator(None)
    c = sim._prepare_circuit(BS())  # Prepare circuit shouldn't change a circuit without any TD
    assert isinstance(c, Circuit)
    assert c.m == 2
    assert len(c._components) == 1
    assert c._components[0][0] == (0,1)
    assert isinstance(c._components[0][1], BS)

    input_circ = [((0,1), BS()), ((0,), TD(1)), ((0,1), BS())]
    output_circ = sim._prepare_circuit(input_circ)
    assert output_circ.m == 5


def test_delay_simulation():
    backend = NaiveBackend()
    simulator = DelaySimulator(Simulator(backend))
    input_circ = [((0, 1), BS()), ((0,), TD(1)), ((0, 1), BS())]
    simulator.set_circuit(input_circ)
    res = simulator.probs(BasicState([1, 0]))

    expected = BSDistribution()
    expected[BasicState([0, 0])] = 0.25
    expected[BasicState([1, 0])] = 0.25
    expected[BasicState([0, 1])] = 0.25
    expected[BasicState([2, 0])] = 0.125
    expected[BasicState([0, 2])] = 0.125

    assert pytest.approx(res) == expected

    simulator.set_selection(postselect=PostSelect('[1]>0'))

    heralded_res = simulator.probs(BasicState([1, 0]))
    expected = BSDistribution()
    expected[BasicState([0, 1])] = 0.666666666666667
    expected[BasicState([0, 2])] = 0.333333333333333
    assert pytest.approx(heralded_res) == expected

    sv = simulator.evolve(BasicState([1, 0]))
    assert len(sv.keys()) == 2
    assert BasicState([0, 1]) in sv.keys()
    assert BasicState([0, 2]) in sv.keys()
