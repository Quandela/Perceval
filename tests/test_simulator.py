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

import math
import pytest

from perceval import catalog
from perceval.backends import AProbAmpliBackend, SLOSBackend
from perceval.simulators import Simulator
from perceval.components import Circuit, BS, PS, Source, unitary_components
from perceval.utils import BasicState, BSDistribution, StateVector, SVDistribution, PostSelect, Matrix, DensityMatrix
from _test_utils import assert_sv_close, assert_svd_close


class MockBackend(AProbAmpliBackend):

    @property
    def name(self) -> str:
        return "Mock"

    def prob_amplitude(self, output_state: BasicState) -> complex:
        return 0

    def prob_distribution(self) -> BSDistribution:
        n = self._input_state.n
        m = self._input_state.m
        output_state = [0]*m
        output_state[(n-1) % m] = n
        return BSDistribution(BasicState(output_state))

    def evolve(self) -> StateVector:
        n = self._input_state.n
        m = self._input_state.m
        output_state = [0] * m
        output_state[(n-1) % m] = n
        return StateVector(output_state)


def test_simulator_probs_mock():
    input_state = BasicState([1,1,1])
    simulator = Simulator(MockBackend())
    simulator.set_circuit(Circuit(3))
    output_dist = simulator.probs(input_state)
    assert len(output_dist) == 1
    assert list(output_dist.keys())[0] == BasicState([0, 0, 3])
    assert simulator.DEBUG_evolve_count == 1

    input_state = BasicState('|{_:1},{_:2},{_:3}>')
    output_dist = simulator.probs(input_state)
    assert len(output_dist) == 1
    assert list(output_dist.keys())[0] == BasicState([3, 0, 0])
    assert simulator.DEBUG_evolve_count == 4

    input_state = BasicState('|{_:1}{_:2}{_:3},0,0>')
    output_dist = simulator.probs(input_state)
    assert len(output_dist) == 1
    assert list(output_dist.keys())[0] == BasicState([3, 0, 0])
    assert simulator.DEBUG_evolve_count == 4


def test_simulator_probs_svd_indistinguishable():
    svd = SVDistribution()
    svd[StateVector([1,0]) + StateVector([0,1])] = 0.3
    svd[StateVector([1,1]) + 1j*StateVector([0,1])] = 0.3
    svd[StateVector('|2,0>') + StateVector([1,1])] = 0.4
    simulator = Simulator(SLOSBackend())
    simulator.set_circuit(BS())
    res = simulator.probs_svd(svd)['results']
    assert len(res) == 5
    assert res[BasicState("|1,0>")] == pytest.approx(0.225)
    assert res[BasicState("|0,1>")] == pytest.approx(0.225)
    assert res[BasicState("|2,0>")] == pytest.approx(0.225)
    assert res[BasicState("|0,2>")] == pytest.approx(0.225)
    assert res[BasicState("|1,1>")] == pytest.approx(0.1)


def test_simulator_probs_svd_distinguishable():
    in_svd = SVDistribution({
        BasicState('|{_:0}{_:1},{_:0}>'): 1
    })
    circuit = BS.H(theta=BS.r_to_theta(0.4))
    sim = Simulator(SLOSBackend())
    sim.set_circuit(circuit)
    res = sim.probs_svd(in_svd)['results']
    assert len(res) == 4
    assert res[BasicState("|3,0>")] == pytest.approx(0.192)
    assert res[BasicState("|2,1>")] == pytest.approx(0.304)
    assert res[BasicState("|1,2>")] == pytest.approx(0.216)
    assert res[BasicState("|0,3>")] == pytest.approx(0.288)


def test_simulator_probs_svd_superposed():
    superposed_state = StateVector("|0,{_:0},{_:1},0>") + StateVector("|0,{_:1},{_:0},0>")
    in_svd = SVDistribution({superposed_state: 1})
    circuit = Circuit(4)
    circuit.add(1, BS.H()).add(0, BS.H(BS.r_to_theta(1/3), phi_tl=-math.pi / 2, phi_bl=math.pi, phi_tr=math.pi / 2))
    circuit.add(2, BS.H(BS.r_to_theta(1/3))).add(1, BS.H())
    sim = Simulator(SLOSBackend())
    sim.set_circuit(circuit)
    res = sim.probs_svd(in_svd)['results']
    assert len(res) == 7
    assert res[BasicState("|2,0,0,0>")] == pytest.approx(2/9)
    assert res[BasicState("|0,0,0,2>")] == pytest.approx(2/9)
    assert res[BasicState("|1,0,1,0>")] == pytest.approx(1/9)
    assert res[BasicState("|1,1,0,0>")] == pytest.approx(1/9)
    assert res[BasicState("|0,1,1,0>")] == pytest.approx(1/9)
    assert res[BasicState("|0,1,0,1>")] == pytest.approx(1/9)
    assert res[BasicState("|0,0,1,1>")] == pytest.approx(1/9)


def test_simulator_probs_distinguishable():
    in_state = BasicState('|{_:0}{_:1},{_:0}>')
    circuit = BS.H(theta=BS.r_to_theta(0.4))
    sim = Simulator(SLOSBackend())
    sim.set_circuit(circuit)
    res = sim.probs(in_state)
    assert len(res) == 4
    assert res[BasicState("|3,0>")] == pytest.approx(0.192)
    assert res[BasicState("|2,1>")] == pytest.approx(0.304)
    assert res[BasicState("|1,2>")] == pytest.approx(0.216)
    assert res[BasicState("|0,3>")] == pytest.approx(0.288)


def test_simulator_probs_postselection():
    input_state = BasicState([1, 1, 1])
    ps = PostSelect("[2] < 2")  # At most 1 photon on mode #2
    simulator = Simulator(MockBackend())
    simulator.set_postselection(ps)
    simulator.set_circuit(Circuit(3))

    with pytest.warns(UserWarning):
        output_dist = simulator.probs(input_state)

    assert len(output_dist) == 0
    assert simulator.logical_perf == pytest.approx(0)

    input_state = BasicState('|{_:1},{_:2},{_:3}>')
    output_dist = simulator.probs(input_state)
    assert len(output_dist) == 1
    assert list(output_dist.keys())[0] == BasicState([3, 0, 0])
    assert simulator.logical_perf == pytest.approx(1)

    input_state = BasicState('|{_:1}{_:2}{_:3},0,0>')
    output_dist = simulator.probs(input_state)
    assert len(output_dist) == 1
    assert list(output_dist.keys())[0] == BasicState([3, 0, 0])
    assert simulator.logical_perf == pytest.approx(1)


def test_simulator_probampli():
    input_state = BasicState("|{_:0},{_:1}>")
    simulator = Simulator(SLOSBackend())
    simulator.set_circuit(BS())
    assert simulator.prob_amplitude(input_state, BasicState("|{_:0}{_:1},0>")) == pytest.approx(0.5j)
    assert simulator.prob_amplitude(input_state, BasicState("|0,{_:0}{_:1}>")) == pytest.approx(0.5j)
    assert simulator.prob_amplitude(input_state, BasicState("|{_:0},{_:1}>")) == pytest.approx(0.5)
    assert simulator.prob_amplitude(input_state, BasicState("|{_:1},{_:0}>")) == pytest.approx(-0.5)
    assert simulator.prob_amplitude(input_state, BasicState("|2,0>")) == pytest.approx(0)
    assert simulator.prob_amplitude(input_state, BasicState("|1,1>")) == pytest.approx(0)
    # prob_amplitude call is strict on annotations name
    assert simulator.prob_amplitude(input_state, BasicState("|{_:0}{_:2},0>")) == pytest.approx(0)

    input_state = StateVector("|{_:0},{_:1}>")
    assert simulator.prob_amplitude(input_state, BasicState("|{_:0}{_:1},0>")) == pytest.approx(0.5j)
    assert simulator.prob_amplitude(input_state, BasicState("|0,{_:0}{_:1}>")) == pytest.approx(0.5j)
    assert simulator.prob_amplitude(input_state, BasicState("|{_:0},{_:1}>")) == pytest.approx(0.5)
    assert simulator.prob_amplitude(input_state, BasicState("|{_:1},{_:0}>")) == pytest.approx(-0.5)
    assert simulator.prob_amplitude(input_state, BasicState("|2,0>")) == pytest.approx(0)
    assert simulator.prob_amplitude(input_state, BasicState("|1,1>")) == pytest.approx(0)
    # prob_amplitude call is strict on annotations name
    assert simulator.prob_amplitude(input_state, BasicState("|{_:0}{_:2},0>")) == pytest.approx(0)

def test_simulator_probability():
    input_state = BasicState("|{_:0},{_:1}>")
    simulator = Simulator(SLOSBackend())
    simulator.set_circuit(BS())
    # Output annotations are ignored for a probability call
    assert simulator.probability(input_state, BasicState("|{_:0}{_:1},0>")) == pytest.approx(0.25)
    assert simulator.probability(input_state, BasicState("|2,0>")) == pytest.approx(0.25)
    assert simulator.probability(input_state, BasicState("|0,2>")) == pytest.approx(0.25)
    assert simulator.probability(input_state, BasicState("|1,1>")) == pytest.approx(0.5)

    input_state = BasicState("|1,1>")
    assert simulator.probability(input_state, BasicState("|{_:0}{_:1},0>")) == pytest.approx(0.5)
    assert simulator.probability(input_state, BasicState("|2,0>")) == pytest.approx(0.5)
    assert simulator.probability(input_state, BasicState("|0,2>")) == pytest.approx(0.5)
    assert simulator.probability(input_state, BasicState("|1,1>")) == pytest.approx(0.0)

    input_state = StateVector("|{_:0},{_:1}>")
    assert simulator.probability(input_state, BasicState("|{_:0}{_:1},0>")) == pytest.approx(0.25)
    assert simulator.probability(input_state, BasicState("|2,0>")) == pytest.approx(0.25)
    assert simulator.probability(input_state, BasicState("|0,2>")) == pytest.approx(0.25)
    assert simulator.probability(input_state, BasicState("|1,1>")) == pytest.approx(0.5)


def test_simulator_probs_sv():
    st1 = StateVector("|0,1>")
    st2 = StateVector("|1,0>")
    sv = st1 + st2
    simulator = Simulator(SLOSBackend())
    c = BS.H()
    simulator.set_circuit(c)
    result = simulator.probs(sv)
    assert len(result) == 1
    assert result[BasicState("|1,0>")] == pytest.approx(1)

    input_state = BasicState("|{_:0},{_:1}>") + BasicState([1, 1])
    simulator.set_circuit(c)
    result = simulator.probs(input_state)
    assert len(result) == 3
    assert result[BasicState("|2,0>")] == pytest.approx(3/8)
    assert result[BasicState("|0,2>")] == pytest.approx(3/8)
    assert result[BasicState("|1,1>")] == pytest.approx(1/4)

    simulator.set_circuit(BS())
    s_boson = StateVector("|{Q:0},{Q:1}>") + StateVector("|{Q:1},{Q:0}>")
    s_fermion = StateVector("|{Q:0},{Q:1}>") - StateVector("|{Q:1},{Q:0}>")
    result_boson = simulator.probs(s_boson)
    assert len(result_boson) == 2
    assert result_boson[BasicState("|2,0>")] == pytest.approx(1/2)
    assert result_boson[BasicState("|0,2>")] == pytest.approx(1/2)
    result_fermion = simulator.probs(s_fermion)
    assert len(result_fermion) == 1
    assert result_fermion[BasicState("|1,1>")] == pytest.approx(1)

    result2_2 = simulator.probs(BasicState("|2,2>"))
    assert len(result2_2) == 3
    assert result2_2[BasicState("|4,0>")] == pytest.approx(0.375)
    assert result2_2[BasicState("|2,2>")] == pytest.approx(0.25)
    assert result2_2[BasicState("|0,4>")] == pytest.approx(0.375)


def test_evolve_indistinguishable():
    simulator = Simulator(SLOSBackend())
    simulator.set_circuit(BS.H())
    sv1 = BasicState([1, 1])
    sv1_out = simulator.evolve(sv1)
    assert_sv_close(sv1_out, math.sqrt(2)/2*StateVector([2, 0]) - math.sqrt(2)/2*StateVector([0, 2]))
    sv1_out_out = simulator.evolve(sv1_out)
    assert_sv_close(sv1_out_out, StateVector([1, 1]))


def test_evolve_distinguishable():
    simulator = Simulator(SLOSBackend())
    simulator.set_circuit(BS.H())
    sv2 = StateVector("|{a:0},{a:0}{a:1}>")
    sv2_out = simulator.evolve(sv2)
    assert pytest.approx(sv2_out[BasicState('|2{a:0}{a:1},0>')]) == 1/2
    assert pytest.approx(sv2_out[BasicState('|2{a:0},{a:1}>')]) == -1/2
    assert pytest.approx(sv2_out[BasicState('|{a:1},2{a:0}>')]) == -1/2
    assert pytest.approx(sv2_out[BasicState('|0,2{a:0}{a:1}>')]) == 1/2
    sv2_out_out = simulator.evolve(sv2_out)
    assert_sv_close(sv2_out_out, sv2)


def test_statevector_polar_evolve():
    simulator = Simulator(SLOSBackend())
    simulator.set_circuit(BS())
    st1 = StateVector("|{P:H},{P:H}>")
    st2 = StateVector("|{P:H},{P:V}>")
    gamma = math.pi / 2
    input_state = math.cos(gamma) * st1 + math.sin(gamma) * st2

    sum_p = sum(list(simulator.probs(input_state).values()))
    assert pytest.approx(1) == sum_p

    sum_p = sum(list(simulator.probs(st2).values()))
    assert pytest.approx(1) == sum_p


def test_evolve_phase():
    input_state = StateVector([2, 0]) + StateVector([1, 1])
    c = Circuit(2).add(1, PS(phi=math.pi/3))
    simu = Simulator(SLOSBackend())
    simu.set_circuit(c)
    output_sv = simu.evolve(input_state)
    assert output_sv[BasicState([1, 1])] == pytest.approx(complex(math.sqrt(2)/4, math.sqrt(6)/4))

    input_state2 = StateVector([0, 0])
    assert simu.evolve(input_state2) == StateVector([0,0])


def test_simulator_evolve_svd():
    input_svd = SVDistribution({StateVector([1, 1]): 0.2,
                                StateVector([2, 0]): 0.8})
    b = SLOSBackend()
    b.set_circuit(Circuit(2).add(0, BS.H()))
    sim = Simulator(b)
    svd_expected = SVDistribution({(math.sqrt(2)/2)*BasicState([2,0])-(math.sqrt(2)/2)*BasicState([0,2]): 0.2,
                                   0.5*BasicState([2,0])+0.5*BasicState([0,2])+(math.sqrt(2)/2)*BasicState([1,1]): 0.8})

    assert_svd_close(sim.evolve_svd(input_svd)['results'], svd_expected)

    ps = PostSelect("[0] == 1")
    sv = BasicState([0, 1]) + BasicState([1, 0])
    sv.normalize()
    input_svd_2 = SVDistribution({sv: 0.2,
                                  StateVector([1,0]): 0.8})
    sim.set_postselection(ps)
    output_svd_2 = sim.evolve_svd(input_svd_2)["results"]
    assert len(output_svd_2) == 1
    assert output_svd_2[StateVector([1,0])] == pytest.approx(1)


def test_heralds():
    sim = Simulator(SLOSBackend())
    sim.set_circuit(catalog['heralded cnot'].build_circuit())
    heralds = {4: 1, 5: 1}
    sim.set_selection(heralds=heralds)

    input_state = StateVector("|0,1,0,1,1,1>") + StateVector("|0,1,1,0,1,1>")
    assert_sv_close(sim.evolve(input_state), input_state)
    sim.keep_heralds(False)
    assert_sv_close(sim.evolve(input_state), StateVector("|0,1,0,1>") + StateVector("|0,1,1,0>"))

    input_state = BasicState("|0,1,0,1,1,1>")
    sim.keep_heralds(True)
    assert_sv_close(sim.evolve(input_state), StateVector("|0,1,1,0,1,1>"))
    sim.keep_heralds(False)
    assert_sv_close(sim.evolve(input_state), StateVector("|0,1,1,0>"))

    s = Source(.9)
    svd = s.generate_distribution(input_state)
    sim.keep_heralds(True)
    keep_heralds_output = sim.evolve_svd(svd)
    sim.keep_heralds(False)
    discard_heralds_output = sim.evolve_svd(svd)

    for kh_state, dh_state in zip(keep_heralds_output['results'].keys(), discard_heralds_output['results'].keys()):
        assert_sv_close(kh_state, dh_state * StateVector([1, 1]))



def get_comparison_setup():
    s = Source(.9)
    svd = s.generate_distribution(BasicState([1, 1, 1, 1]))
    U = Matrix.random_unitary(4)
    circuit = Circuit(4).add(0, unitary_components.Unitary(U))
    sim = Simulator(SLOSBackend())
    sim.set_circuit(circuit)
    dm = DensityMatrix.from_svd(svd)
    return sim, dm, svd


def test_evolve_density_matrix():

    sim, dm, svd = get_comparison_setup()
    final_svd = sim.evolve_svd(svd)["results"]
    final_dm = sim.evolve_density_matrix(dm)
    comparing_dm = DensityMatrix.from_svd(final_svd)

    assert max((final_dm.mat-comparing_dm.mat).data) < 1e-10


def test_probs_density_matrix():

    sim, dm, svd = get_comparison_setup()

    probs_1 = sim.probs_svd(svd)["results"]
    probs_2 = sim.probs_density_matrix(dm)["results"]

    for key, value in probs_1.items():
        assert probs_2[key] == pytest.approx(value)

    for key, value in probs_2.items():
        assert probs_1[key] == pytest.approx(value)
