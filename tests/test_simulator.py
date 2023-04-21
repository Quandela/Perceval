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

from perceval.backends._abstract_backends import AProbAmpliBackend
from perceval.backends._naive import NaiveBackend
from perceval.backends.simulator import Simulator
from perceval.components import Circuit, BS
from perceval.utils import BasicState, BSDistribution, StateVector


class MockBackend(AProbAmpliBackend):
    def prob_amplitude(self, output_state: BasicState) -> complex:
        return 0

    def prob_distribution(self) -> BSDistribution:
        n = self._input_state.n
        m = self._input_state.m
        output_state = [0]*m
        output_state[(n-1) % m] = n
        return BSDistribution(BasicState(output_state))


def test_simulator_probs():
    input_state = BasicState([1,1,1])
    simulator = Simulator(MockBackend())
    simulator.set_circuit(Circuit(3))
    output_dist = simulator.probs(input_state)
    assert len(output_dist) == 1
    assert list(output_dist.keys())[0] == BasicState([0, 0, 3])
    assert simulator.DEBUG_computation_count == 1

    input_state = BasicState('|{_:1},{_:2},{_:3}>')
    output_dist = simulator.probs(input_state)
    assert len(output_dist) == 1
    assert list(output_dist.keys())[0] == BasicState([3, 0, 0])
    assert simulator.DEBUG_computation_count == 3

    input_state = BasicState('|{_:1}{_:2}{_:3},0,0>')
    output_dist = simulator.probs(input_state)
    assert len(output_dist) == 1
    assert list(output_dist.keys())[0] == BasicState([3, 0, 0])
    assert simulator.DEBUG_computation_count == 1


def test_evolve_indistinguishable():
    simulator = Simulator(NaiveBackend())
    simulator.set_circuit(BS.H())
    sv1 = BasicState([1, 1])
    sv1_out = simulator.evolve(sv1)
    assert str(sv1_out) == "sqrt(2)/2*|2,0>-sqrt(2)/2*|0,2>"
    sv1_out_out = simulator.evolve(sv1_out)
    assert str(sv1_out_out) == "|1,1>"


def test_evolve_distinguishable():
    simulator = Simulator(NaiveBackend())
    simulator.set_circuit(BS.H())
    sv2 = BasicState("|{a:0},{a:0}{a:1}>")
    sv2_out = simulator.evolve(sv2)
    assert str(sv2_out) == "1/2*|2{a:0}{a:1},0>-1/2*|2{a:0},{a:1}>-1/2*|{a:1},2{a:0}>+1/2*|0,2{a:0}{a:1}>"
    sv2_out_out = simulator.evolve(sv2_out)
    assert str(sv2_out_out) == "|{a:0},{a:0}{a:1}>"
