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
import warnings
from math import sqrt

from perceval.backends import Clifford2017Backend, NaiveBackend, AProbAmpliBackend, SLOSBackend, MPSBackend,\
    BackendFactory
from perceval.components import BS, PS, Circuit
from perceval.utils import BSCount, BasicState, Parameter, StateVector
import pytest
import numpy as np


def _cnot_circuit():
    theta_13 = BS.r_to_theta(1 / 3)
    cnot = (Circuit(6, name="CNOT")
            .add((0, 1), BS.H(theta_13, phi_bl=np.pi, phi_tr=np.pi / 2, phi_tl=-np.pi / 2))
            .add((3, 4), BS.H())
            .add((2, 3), BS.H(theta_13, phi_bl=np.pi, phi_tr=np.pi / 2, phi_tl=-np.pi / 2))
            .add((4, 5), BS.H(theta_13))
            .add((3, 4), BS.H()))
    return cnot


def _assert_cnot(backend: AProbAmpliBackend):
    backend.set_input_state(BasicState([0, 1, 0, 1, 0, 0]))
    assert pytest.approx(backend.probability(BasicState([0, 1, 0, 1, 0, 0]))) == 1 / 9
    assert pytest.approx(backend.probability(BasicState([0, 1, 0, 0, 1, 0]))) == 0
    backend.set_input_state(BasicState([0, 1, 0, 0, 1, 0]))
    assert pytest.approx(backend.probability(BasicState([0, 1, 0, 0, 1, 0]))) == 1 / 9
    assert pytest.approx(backend.probability(BasicState([0, 1, 0, 1, 0, 0]))) == 0
    backend.set_input_state(BasicState([0, 0, 1, 1, 0, 0]))
    assert pytest.approx(backend.probability(BasicState([0, 0, 1, 0, 1, 0]))) == 1 / 9
    assert pytest.approx(backend.probability(BasicState([0, 0, 1, 1, 0, 0]))) == 0
    backend.set_input_state(BasicState([0, 0, 1, 0, 1, 0]))
    assert pytest.approx(backend.probability(BasicState([0, 0, 1, 0, 1, 0]))) == 0
    assert pytest.approx(backend.probability(BasicState([0, 0, 1, 1, 0, 0]))) == 1 / 9



def test_clifford_bs():
    cliff_bs = Clifford2017Backend()
    cliff_bs.set_circuit(BS.H())
    cliff_bs.set_input_state(BasicState([0, 1]))
    counts = BSCount()
    for _ in range(10000):
        counts[cliff_bs.sample()] += 1
    assert 4750 < counts[BasicState("|0,1>")] < 5250
    assert 4750 < counts[BasicState("|1,0>")] < 5250


def check_output_distribution(backend: AProbAmpliBackend, input_state: BasicState, expected: dict):
    backend.set_input_state(input_state)
    prob_list = []
    for (output_state, prob) in backend.prob_distribution().items():
        prob_expected = expected.get(output_state)
        if prob_expected is None:
            assert pytest.approx(0) == prob, "cannot find: %s (prob=%f)" % (str(output_state), prob)
        else:
            assert pytest.approx(prob_expected) == prob,\
                "incorrect value for %s: %f/%f" % (str(output_state), prob, prob_expected)
        prob_list.append(prob)
    assert pytest.approx(sum(prob_list)) == 1


def test_backend_factory_default():
    default_backend = BackendFactory.get_backend()
    default_backend.set_circuit(BS.H())
    check_output_distribution(default_backend, BasicState([1, 0]),
                              {BasicState("|1,0>"): 0.5, BasicState("|0,1>"): 0.5})


def test_backend_identity():
    for backend_name in ["SLOS", "Naive"]:
        backend = BackendFactory.get_backend(backend_name)
        backend.set_circuit(Circuit(2))  # Identity circuit, 2 modes
        check_output_distribution(backend, BasicState([0, 0]), {BasicState("|0,0>"): 1})
        check_output_distribution(backend, BasicState([0, 1]), {BasicState("|0,1>"): 1})
        check_output_distribution(backend, BasicState([1, 1]), {BasicState("|1,1>"): 1})


def test_backend_wrong_size():
    circuit = Circuit(2)
    state = BasicState([1, 1, 1])
    for backend_name in ["SLOS", "Naive", "MPS", "CliffordClifford2017"]:
        backend = BackendFactory.get_backend(backend_name)
        with pytest.raises(AssertionError):
            backend.set_circuit(circuit)
            backend.set_input_state(state)


def test_backend_sym_bs():
    for backend_name in ["SLOS", "Naive"]:
        backend = BackendFactory.get_backend(backend_name)
        backend.set_circuit(BS.H())
        check_output_distribution(backend, BasicState("|2,0>"),
                                  {BasicState("|2,0>"): 0.25,
                                   BasicState("|1,1>"): 0.5,
                                   BasicState("|0,2>"): 0.25})
        check_output_distribution(backend, BasicState("|1,0>"),
                                  {BasicState("|1,0>"): 0.5,
                                   BasicState("|0,1>"): 0.5})
        check_output_distribution(backend, BasicState("|1,1>"),
                                  {BasicState("|2,0>"): 0.5,
                                   BasicState("|0,2>"): 0.5})


def test_backend_asym_bs():
    for backend_name in ["SLOS", "Naive"]:
        backend = BackendFactory.get_backend(backend_name)
        backend.set_circuit(BS.H(theta=2*np.pi/3))
        check_output_distribution(backend, BasicState("|2,0>"),
                                  {BasicState("|2,0>"): 0.0625,
                                   BasicState("|1,1>"): 0.3750,
                                   BasicState("|0,2>"): 0.5625})
        check_output_distribution(backend, BasicState("|1,0>"),
                                  {BasicState("|1,0>"): 0.25,
                                   BasicState("|0,1>"): 0.75})


def test_slos_precomputation():
    """ Check if the SLOS backend is keeping internal structure"""
    slos = SLOSBackend()
    slos.set_circuit(Circuit(2))
    slos.set_input_state(BasicState([0, 1]))
    assert len(slos._fsms) == 2
    fsm_precompute_1 = slos._fsms[-1]
    slos.set_input_state(BasicState([1, 0]))
    assert len(slos._fsms) == 2
    assert fsm_precompute_1 is slos._fsms[-1]
    slos.set_input_state(BasicState([1, 1]))
    assert len(slos._fsms) == 3
    assert fsm_precompute_1 is slos._fsms[1]


def test_slos_symbolic():
    slos = SLOSBackend(use_symbolic=True)
    c = BS.H(theta=Parameter("theta"))
    slos.set_circuit(c)
    slos.set_input_state(BasicState([0, 1]))
    assert str(slos.probability(BasicState([0, 1]))) == "1.0*cos(theta/2)**2"
    slos.set_input_state(BasicState([1, 0]))
    assert str(slos.probability(BasicState([0, 1]))) == "1.0*sin(theta/2)**2"


def test_backend_cnot():
    for backend_name in ["SLOS", "Naive", "MPS"]:
        backend = BackendFactory.get_backend(backend_name)
        cnot = _cnot_circuit()
        backend.set_circuit(cnot)
        _assert_cnot(backend)
        non_post_selected_probability = 0
        backend.set_input_state(BasicState([0, 1, 0, 1, 0, 0]))
        for output_state, prob in backend.prob_distribution().items():
            if output_state[0] or output_state[5]:
                non_post_selected_probability += prob
        assert pytest.approx(non_post_selected_probability) == 7/9


def test_slos_cnot_with_mask():
    slos_cnot = SLOSBackend(n=2, mask=["0    0"])
    cnot = _cnot_circuit()
    slos_cnot.set_circuit(cnot)
    _assert_cnot(slos_cnot)
    non_post_selected_probability = 0
    slos_cnot.set_input_state(BasicState([0, 1, 0, 1, 0, 0]))
    for output_state, prob in slos_cnot.prob_distribution().items():
        if output_state[0] or output_state[5]:
            non_post_selected_probability += prob
    assert pytest.approx(non_post_selected_probability) == 0


def test_probampli_backends():
    for backend_type in [NaiveBackend, SLOSBackend, MPSBackend]:
        backend = backend_type()
        circuit = Circuit(3) // BS.H() // (1, PS(np.pi/4)) // (1, BS.H())
        backend.set_circuit(circuit)
        check_output_distribution(
            backend,
            BasicState("|0,1,1>"),
            {
                BasicState("|0,1,1>"): 0,
                BasicState("|1,1,0>"): 0.25,
                BasicState("|1,0,1>"): 0.25,
                BasicState("|2,0,0>"): 0,
                BasicState("|0,2,0>"): 0.25,
                BasicState("|0,0,2>"): 0.25,
            })

        if backend_type == MPSBackend:
            warnings.warn("MPS backend is currently broken for input states with multiple photons per mode")
            continue

        backend.set_circuit(BS())
        check_output_distribution(
            backend,
            BasicState("|2,3>"),
            {
                BasicState("|5,0>"): 0.3125,
                BasicState("|4,1>"): 0.0625,
                BasicState("|3,2>"): 0.125,
                BasicState("|2,3>"): 0.125,
                BasicState("|1,4>"): 0.0625,
                BasicState("|0,5>"): 0.3125,
            })


def test_slos_refresh_coefs():
    """
    The previous SLOS implementation was failing to refresh its results when the circuit
    was changed to another circuit of the same size, AFTER multiple fock states were used
    as input.
    The following code reproduces such a behavior
    """
    slos = SLOSBackend()
    slos.set_circuit(BS())  # Use a beam splitter as circuit
    slos.set_input_state(BasicState("|1,1>"))
    slos.set_input_state(BasicState("|8,5>"))
    check_output_distribution(
        slos,
        BasicState("|1,1>"),  # Input
        {  # Expected results for a beam splitter
            BasicState("|0,2>"): 0.5,
            BasicState("|2,0>"): 0.5
        })

    slos.set_circuit(Circuit(2))  # Set the circuit as identity
    check_output_distribution(
        slos,
        BasicState("|1,1>"),
        {  # Expected results for an identity
            BasicState("|1,1>"): 1
        })


def test_evolve_indistinguishable():
    for backend_name in ["SLOS", "Naive", "MPS"]:
        backend = BackendFactory.get_backend(backend_name)
        backend.set_circuit(BS.H())
        backend.set_input_state(BasicState([1, 1]))
        sv_out = backend.evolve()
        assert pytest.approx(sv_out) == sqrt(2)/2*StateVector([2, 0]) - sqrt(2)/2*StateVector([0, 2])
