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

from collections import defaultdict
import pytest
import perceval as pcvl
from perceval.components.unitary_components import BS, PS, Unitary, HWP, WP, PBS
import sympy as sp
import numpy as np


def cnot_circuit():
    theta_13 = BS.r_to_theta(1 / 3)
    cnot = (pcvl.Circuit(6, name="CNOT")
            .add((0, 1), BS.H(theta_13, phi_bl=np.pi, phi_tr=np.pi / 2, phi_tl=-np.pi / 2))
            .add((3, 4), BS.H())
            .add((2, 3), BS.H(theta_13, phi_bl=np.pi, phi_tr=np.pi / 2, phi_tl=-np.pi / 2))
            .add((4, 5), BS.H(theta_13))
            .add((3, 4), BS.H()))
    return cnot


def assert_cnot(s_cnot):
    assert pytest.approx(s_cnot.prob(pcvl.BasicState([0, 1, 0, 1, 0, 0]), pcvl.BasicState([0, 1, 0, 1, 0, 0]))) == 1 / 9
    assert pytest.approx(s_cnot.prob(pcvl.BasicState([0, 1, 0, 0, 1, 0]), pcvl.BasicState([0, 1, 0, 0, 1, 0]))) == 1 / 9
    assert pytest.approx(s_cnot.prob(pcvl.BasicState([0, 1, 0, 1, 0, 0]), pcvl.BasicState([0, 1, 0, 0, 1, 0]))) == 0
    assert pytest.approx(s_cnot.prob(pcvl.BasicState([0, 1, 0, 0, 1, 0]), pcvl.BasicState([0, 1, 0, 1, 0, 0]))) == 0
    assert pytest.approx(s_cnot.prob(pcvl.BasicState([0, 0, 1, 1, 0, 0]), pcvl.BasicState([0, 0, 1, 1, 0, 0]))) == 0
    assert pytest.approx(s_cnot.prob(pcvl.BasicState([0, 0, 1, 0, 1, 0]), pcvl.BasicState([0, 0, 1, 0, 1, 0]))) == 0
    assert pytest.approx(s_cnot.prob(pcvl.BasicState([0, 0, 1, 1, 0, 0]), pcvl.BasicState([0, 0, 1, 0, 1, 0]))) == 1 / 9
    assert pytest.approx(s_cnot.prob(pcvl.BasicState([0, 0, 1, 0, 1, 0]), pcvl.BasicState([0, 0, 1, 1, 0, 0]))) == 1 / 9


def check_output(simulator, input_state, expected):
    all_prob = 0
    for (output_state, prob) in simulator.allstateprob_iterator(input_state):
        prob_expected = expected.get(output_state)
        if prob_expected is None:
            assert pytest.approx(0) == prob, "cannot find: %s (prob=%f)" % (str(output_state), prob)
        else:
            assert pytest.approx(prob_expected) == prob, "incorrect value for %s: %f/%f" % (str(output_state),
                                                                                            prob,
                                                                                            prob_expected)
        all_prob += prob
    assert pytest.approx(all_prob) == 1


def test_minimal():
    # default simulator backend
    simulator_backend = pcvl.BackendFactory().get_backend()
    # simulator directly initialized on circuit
    s = simulator_backend(BS.H())
    check_output(s, pcvl.BasicState([1, 0]), {pcvl.BasicState("|1,0>"): 0.50,
                                              pcvl.BasicState("|0,1>"): 0.50})


def test_building_sim():
    for backend in ["SLOS", "Naive"]:
        simulator_backend = pcvl.BackendFactory().get_backend(backend)
        u = [[1, 0], [0, 1]]
        s = simulator_backend(pcvl.Matrix(u))
        check_output(s, pcvl.BasicState([0, 0]), {pcvl.BasicState("|0,0>"): 1})
        check_output(s, pcvl.BasicState([0, 1]), {pcvl.BasicState("|0,1>"): 1})
        check_output(s, pcvl.BasicState([1, 1]), {pcvl.BasicState("|1,1>"): 1})


def test_sim_indistinct_sym():
    for backend in ["SLOS", "Naive"]:
        simulator_backend = pcvl.BackendFactory().get_backend(backend)
        c = BS.H()
        s = simulator_backend(c.U, use_symbolic=False)
        check_output(s, pcvl.BasicState([2, 0]), {pcvl.BasicState("|2,0>"): 0.25,
                                                  pcvl.BasicState("|1,1>"): 0.50,
                                                  pcvl.BasicState("|0,2>"): 0.25})
        check_output(s, pcvl.BasicState([1, 0]), {pcvl.BasicState("|1,0>"): 0.50,
                                                  pcvl.BasicState("|0,1>"): 0.50})


def test_sim_indistinct_asym():
    for backend in ["SLOS", "Naive"]:
        simulator_backend = pcvl.BackendFactory().get_backend(backend)
        c = BS.H(theta=2*sp.pi/3)
        s = simulator_backend(c.U, use_symbolic=False)
        check_output(s, pcvl.BasicState([2, 0]), {pcvl.BasicState("|2,0>"): 0.0625,
                                                  pcvl.BasicState("|1,1>"): 0.3750,
                                                  pcvl.BasicState("|0,2>"): 0.5625})
        check_output(s, pcvl.BasicState([1, 0]), {pcvl.BasicState("|1,0>"): 0.250,
                                                  pcvl.BasicState("|0,1>"): 0.750})


def test_sim_indistinct_sym11():
    for backend in ["SLOS", "Naive"]:
        simulator_backend = pcvl.BackendFactory().get_backend(backend)
        c = BS.H()
        s = simulator_backend(c, use_symbolic=False)
        check_output(s, pcvl.BasicState([1, 1]), {pcvl.BasicState("|2,0>"): 0.5000,
                                                  pcvl.BasicState("|0,2>"): 0.5000})


def test_check_precompute():
    """ Check if the SLOS backend is keeping internal structure
    """
    simulator_backend = pcvl.BackendFactory().get_backend("SLOS")
    u = [[1, 0], [0, 1]]
    simulator = simulator_backend(pcvl.Matrix(u))
    simulator.sample(pcvl.BasicState([0, 1]))
    assert len(simulator.fsms) == 2
    fsm_precompute_1 = simulator.fsms[-1]
    simulator.sample(pcvl.BasicState([1, 0]))
    assert len(simulator.fsms) == 2
    assert fsm_precompute_1 is simulator.fsms[-1]
    simulator.sample(pcvl.BasicState([1, 1]))
    assert len(simulator.fsms) == 3
    assert fsm_precompute_1 is simulator.fsms[1]


def test_symbolic_prob():
    simulator_backend = pcvl.BackendFactory().get_backend("SLOS")
    c = BS.H(theta=pcvl.Parameter("theta"))
    s = simulator_backend(c.U)
    assert str(s.prob(pcvl.BasicState([0, 1]), pcvl.BasicState([0, 1]))) == "cos(theta/2)**2"
    assert str(s.prob(pcvl.BasicState([1, 0]), pcvl.BasicState([0, 1]))) == "sin(theta/2)**2"


def test_cnot_no_mask():
    for backend in ["SLOS", "Naive"]:
        simulator_backend = pcvl.BackendFactory().get_backend(backend)
        cnot = cnot_circuit()
        s_cnot = simulator_backend(cnot, use_symbolic=False)
        assert_cnot(s_cnot)
        non_post_selected_probability = 0
        for (output_state, prob) in s_cnot.allstateprob_iterator(pcvl.BasicState([0, 1, 0, 1, 0, 0])):
            if output_state[0] or output_state[5]:
                non_post_selected_probability += prob
        assert pytest.approx(non_post_selected_probability) == 7/9


def test_cnot_with_mask():
    for backend in ["SLOS", "Naive"]:
        simulator_backend = pcvl.BackendFactory().get_backend(backend)
        cnot = cnot_circuit()
        s_cnot = simulator_backend(cnot.U, use_symbolic=False, n=2, mask=["0    0"])
        assert_cnot(s_cnot)
        non_post_selected_probability = 0
        for (output_state, prob) in s_cnot.allstateprob_iterator(pcvl.BasicState([0, 1, 0, 1, 0, 0])):
            if output_state[0] or output_state[5]:
                non_post_selected_probability += prob
        assert pytest.approx(non_post_selected_probability) == 0


def test_compile():
    simulator_backend = pcvl.BackendFactory().get_backend("SLOS")
    cnot = cnot_circuit()
    s_cnot = simulator_backend(cnot.U, use_symbolic=False, n=2, mask=["0    0"])
    assert s_cnot.compile([pcvl.BasicState([0, 1, 0, 1, 0, 0]),
                           pcvl.BasicState([0, 0, 1, 1, 0, 0]),
                           pcvl.BasicState([0, 1, 0, 0, 1, 0]),
                           pcvl.BasicState([0, 0, 1, 0, 1, 0])])
    assert_cnot(s_cnot)
    assert not s_cnot.compile([pcvl.BasicState([0, 0, 1, 1, 0, 0])])


def test_non_symmetrical():
    for backend in ["Naive", "SLOS"]:
        # default simulator backend
        simulator_backend = pcvl.BackendFactory().get_backend(backend)
        # simulator directly initialized on circuit
        circuit = pcvl.Circuit(3)
        circuit.add((0, 1), BS.H())
        circuit.add((1,), PS(sp.pi/4))
        circuit.add((1, 2), BS.H())
        pcvl.pdisplay(circuit.U)
        s = simulator_backend(circuit.U)
        check_output(s, pcvl.BasicState([0, 1, 1]), {pcvl.BasicState("|0,1,1>"): 0,
                                                     pcvl.BasicState("|1,1,0>"): 0.25,
                                                     pcvl.BasicState("|1,0,1>"): 0.25,
                                                     pcvl.BasicState("|2,0,0>"): 0,
                                                     pcvl.BasicState("|0,2,0>"): 0.25,
                                                     pcvl.BasicState("|0,0,2>"): 0.25,
                                                     })


def test_evolve_indistinguishable():
    c = BS.H()
    for backend_name in ["SLOS", "Naive"]:
        simulator = pcvl.BackendFactory().get_backend(backend_name)(c)
        sv1 = pcvl.StateVector([1, 1])
        check_output(simulator, sv1, {pcvl.BasicState("|0,2>"): 0.5, pcvl.BasicState("|2,0>"): 0.5})
        sv1_out = simulator.evolve(sv1)
        assert str(sv1_out) == "sqrt(2)/2*|2,0>-sqrt(2)/2*|0,2>"
        sv2 = pcvl.StateVector([1, 1], {0: ["_:0"]})
        check_output(simulator, sv2, {pcvl.BasicState("|0,2>"): 0.5, pcvl.BasicState("|2,0>"): 0.5})
        sv3 = pcvl.StateVector([1, 1], {0: ["_:0"], 1: ["_:0"]})
        check_output(simulator, sv3, {pcvl.BasicState("|0,2>"): 0.5, pcvl.BasicState("|2,0>"): 0.5})


def test_hybrid_state():
    c = BS.H()
    for backend_name in ["SLOS", "Naive"]:
        simulator = pcvl.BackendFactory().get_backend(backend_name)(c)
        sv1 = pcvl.StateVector([1, 1], {0: ["_:1"], 1: ["_:2"]})
        check_output(simulator, sv1, {pcvl.BasicState("|0,2>"): 0.25,
                                      pcvl.BasicState("|2,0>"): 0.25,
                                      pcvl.BasicState("|1,1>"): 0.5})


def test_state_entanglement():
    st1 = pcvl.StateVector("|0,1>")
    st2 = pcvl.StateVector("|1,0>")
    st3 = st1+st2
    c = BS.H()
    for backend_name in ["SLOS", "Naive"]:
        simulator = pcvl.BackendFactory().get_backend(backend_name)(c)
        st3_out = simulator.evolve(st3)
        assert str(st3_out) == "|1,0>"
        check_output(simulator, st3, {pcvl.BasicState("|1,0>"): 1})


def test_clifford_bs():
    bs_backend = pcvl.BackendFactory().get_backend("CliffordClifford2017")
    sim = bs_backend(BS.H())
    counts = defaultdict(int)
    for _ in range(10000):
        counts[sim.sample(pcvl.BasicState([0, 1]))] += 1
    assert 4750 < counts[pcvl.BasicState("|0,1>")] < 5250
    assert 4750 < counts[pcvl.BasicState("|1,0>")] < 5250


def _run_clifford(n: int, m: int):
    state = pcvl.BasicState([1] * n + [0] * (m - n))
    bs_backend = pcvl.BackendFactory().get_backend("CliffordClifford2017")
    u = pcvl.Matrix.random_unitary(m)
    experiment = bs_backend(Unitary(U=u))
    experiment.sample(state)


def test_clifford_10():
    _run_clifford(10, 60)


@pytest.mark.long_test
def test_clifford_27():
    _run_clifford(27, 60)


def test_polarization_circuit_0():
    c = pcvl.Circuit(1)
    c //= HWP(sp.pi/4)
    for backend_name in ["Naive", "SLOS"]:
        simulator = pcvl.BackendFactory().get_backend(backend_name)(c)
        check_output(simulator,
                     pcvl.BasicState("|{P:H}>"),
                     {pcvl.BasicState("|1>"): 1})
        assert pytest.approx(simulator.prob(pcvl.BasicState("|{P:H}>"),
                                            pcvl.BasicState("|{P:H}>"))) == 0
        assert pytest.approx(simulator.prob(pcvl.BasicState("|{P:H}>"),
                                            pcvl.BasicState("|{P:V}>"))) == 1
        assert pytest.approx(simulator.prob(pcvl.BasicState("|{P:V}>"),
                                            pcvl.BasicState("|{P:H}>"))) == 1
        assert pytest.approx(simulator.prob(pcvl.BasicState("|{P:D}>"),
                                            pcvl.BasicState("|{P:D}>"))) == 1
        assert pytest.approx(simulator.prob(pcvl.BasicState("|{P:A}>"),
                                            pcvl.BasicState("|{P:A}>"))) == 1


def test_polarization_circuit_1():
    c = pcvl.Circuit(1)
    c //= WP(sp.pi/2, sp.pi/8)
    for backend_name in ["SLOS", "Naive"]:
        simulator = pcvl.BackendFactory().get_backend(backend_name)(c)
        check_output(simulator,
                     pcvl.BasicState("|{P:H}>"),
                     {pcvl.BasicState("|1>"): 1})
        assert pytest.approx(1) == simulator.prob(pcvl.BasicState("|{P:H}>"), pcvl.BasicState("|{P:D}>"))
        assert pytest.approx(1) == simulator.prob(pcvl.BasicState("|{P:V}>"), pcvl.BasicState("|{P:A}>"))


def test_polarization_circuit_2():
    c = pcvl.Circuit(1)
    c //= WP(sp.pi/4, sp.pi/4)
    for backend_name in ["SLOS", "Naive"]:
        simulator = pcvl.BackendFactory().get_backend(backend_name)(c)
        # check_output(simulator,
        #              pcvl.BasicState("|{P:H}>"),
        #              {pcvl.BasicState("|1>"): 1})
        assert pytest.approx(1) == simulator.prob(pcvl.BasicState("|{P:H}>"), pcvl.BasicState("|{P:L}>"))
        assert pytest.approx(1) == simulator.prob(pcvl.BasicState("|{P:V}>"), pcvl.BasicState("|{P:R}>"))


def test_polarization_circuit_3():
    c = pcvl.Circuit(2) // PBS()
    for backend_name in ["Naive", "SLOS"]:
        simulator = pcvl.BackendFactory().get_backend(backend_name)(c)
        check_output(simulator,
                     pcvl.BasicState("|1,0>"),
                     {pcvl.BasicState("|0,1>"): 1})
        assert pytest.approx(1) == simulator.prob(pcvl.BasicState("|{P:H},0>"), pcvl.BasicState("|0,{P:H}>"))
        assert pytest.approx(1) == simulator.prob(pcvl.BasicState("|{P:V},0>"), pcvl.BasicState("|{P:V},0>"))


@pytest.mark.skip(reason="cannot map multiple polarization to one single-mode")
def test_polarization_circuit_4():
    c = pcvl.Circuit(2) // PBS()
    for backend_name in ["Naive", "SLOS"]:
        simulator = pcvl.BackendFactory().get_backend(backend_name)(c)
        check_output(simulator,
                     pcvl.BasicState("|1,0>"),
                     {pcvl.BasicState("|0,1>"): 1})
        assert simulator.prob(pcvl.BasicState("|{P:H},{P:V}>"),
                              pcvl.BasicState("|0,{P:H}{P:V}>")) == 1


def test_bs_polarization():
    c = BS.H()
    sim = pcvl.BackendFactory().get_backend("Naive")(c)

    input_state = pcvl.BasicState("|{P:V},0>")

    states = [(pcvl.BasicState("|0,{P:H}>"), 0),
              (pcvl.BasicState("|{P:V},0>"), 1/2),
              (pcvl.BasicState("|0,{P:V}>"), 1/2),
              (pcvl.BasicState("|{P:H},0>"), 0),
              (pcvl.BasicState("|{P:V},0>"), 1/2)]

    for output_state, prob in states:
        assert pytest.approx(sim.prob(input_state, output_state)) == prob


def test_all_prob():
    c = BS.H()
    sim = pcvl.BackendFactory().get_backend("Naive")(c)
    assert pytest.approx(np.asarray([0.5, 0, 0.5])) == sim.all_prob(pcvl.BasicState("|1,1>"))
