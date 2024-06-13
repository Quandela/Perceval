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

from perceval.backends import Clifford2017Backend, AProbAmpliBackend, SLOSBackend, BackendFactory
from perceval.components import BS, PS, PERM, Circuit, catalog
from perceval.utils import BSCount, BasicState, Parameter, StateVector
from _test_utils import assert_sv_close


def _assert_cnot(backend: AProbAmpliBackend):
    s00 = BasicState([1, 0, 1, 0, 0, 0])
    s01 = BasicState([1, 0, 0, 1, 0, 0])
    s10 = BasicState([0, 1, 1, 0, 0, 0])
    s11 = BasicState([0, 1, 0, 1, 0, 0])
    backend.set_input_state(s00)
    assert pytest.approx(backend.probability(s00)) == 1 / 9
    assert pytest.approx(backend.probability(s01)) == 0
    backend.set_input_state(s01)
    assert pytest.approx(backend.probability(s01)) == 1 / 9
    assert pytest.approx(backend.probability(s00)) == 0
    backend.set_input_state(s10)
    assert pytest.approx(backend.probability(s11)) == 1 / 9
    assert pytest.approx(backend.probability(s10)) == 0
    backend.set_input_state(s11)
    assert pytest.approx(backend.probability(s11)) == 0
    assert pytest.approx(backend.probability(s10)) == 1 / 9


def test_clifford_bs():
    cliff_bs = Clifford2017Backend()
    cliff_bs.set_circuit(BS.H())
    cliff_bs.set_input_state(BasicState([0, 1]))
    counts = BSCount()
    n_samples = 10000
    for s in cliff_bs.samples(n_samples):
        counts[s] += 1
    assert n_samples*0.475 < counts[BasicState("|0,1>")] < n_samples*0.525
    assert n_samples*0.475 < counts[BasicState("|1,0>")] < n_samples*0.525


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


@pytest.mark.parametrize("backend_name", ["SLOS", "Naive", "NaiveApprox"])
def test_backend_wiring(backend_name):
    backend: AProbAmpliBackend = BackendFactory.get_backend(backend_name)
    backend.set_circuit(Circuit(1))  # Identity circuit, 2 modes
    check_output_distribution(backend, BasicState([1]), {BasicState("|1>"): 1})

@pytest.mark.parametrize("backend_name", ["SLOS", "Naive", "MPS"])
def test_backend_identity(backend_name):
    backend: AProbAmpliBackend = BackendFactory.get_backend(backend_name)
    backend.set_circuit(Circuit(2))  # Identity circuit, 2 modes
    check_output_distribution(backend, BasicState([0, 0]), {BasicState("|0,0>"): 1})
    check_output_distribution(backend, BasicState([0, 1]), {BasicState("|0,1>"): 1})
    check_output_distribution(backend, BasicState([1, 1]), {BasicState("|1,1>"): 1})


@pytest.mark.parametrize("backend_name", ["SLOS", "Naive", "MPS", "CliffordClifford2017"])
def test_backend_wrong_size(backend_name):
    circuit = Circuit(2)
    state = BasicState([1, 1, 1])
    backend = BackendFactory.get_backend(backend_name)
    with pytest.raises(AssertionError):
        backend.set_circuit(circuit)
        backend.set_input_state(state)


@pytest.mark.parametrize("backend_name", ["SLOS", "Naive", "MPS"])
def test_backend_sym_bs(backend_name):
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


@pytest.mark.parametrize("backend_name", ["SLOS", "Naive", "MPS"])
def test_backend_asym_bs(backend_name):
    backend = BackendFactory.get_backend(backend_name)
    backend.set_circuit(BS.H(theta=2*math.pi/3))
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


@pytest.mark.parametrize("backend_name", ["SLOS", "Naive"])  # MPS not working with >2-modes components
def test_backend_cnot(backend_name):
    backend: AProbAmpliBackend = BackendFactory.get_backend(backend_name)
    cnot = catalog["postprocessed cnot"].build_circuit()
    backend.set_circuit(cnot)
    _assert_cnot(backend)
    non_post_selected_probability = 0
    backend.set_input_state(BasicState([1, 0, 1, 0, 0, 0]))  # Two last modes are ancillaries
    for output_state, prob in backend.prob_distribution().items():
        if output_state[4] or output_state[5]:
            non_post_selected_probability += prob
    assert pytest.approx(non_post_selected_probability) == 7/9


def test_slos_cnot_with_mask():
    slos_cnot = SLOSBackend(n=2, mask=["    00"])  # Masking ancillary modes
    cnot = catalog["postprocessed cnot"].build_circuit()
    slos_cnot.set_circuit(cnot)
    _assert_cnot(slos_cnot)
    non_post_selected_probability = 0
    slos_cnot.set_input_state(BasicState([0, 1, 0, 1, 0, 0]))
    for output_state, prob in slos_cnot.prob_distribution().items():
        if output_state[4] or output_state[5]:
            non_post_selected_probability += prob
    assert pytest.approx(non_post_selected_probability) == 0


@pytest.mark.parametrize("backend_name", ["SLOS", "Naive", "MPS"])
def test_probampli_backends(backend_name):
    backend: AProbAmpliBackend = BackendFactory.get_backend(backend_name)
    circuit = Circuit(3) // BS.H() // (1, PS(math.pi/4)) // (1, BS.H())
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

@pytest.mark.parametrize("backend_name", ["SLOS", "Naive", "MPS"])
def test_evolve_indistinguishable(backend_name):
    backend = BackendFactory.get_backend(backend_name)
    backend.set_circuit(BS.H())
    backend.set_input_state(BasicState([1, 0]))
    sv_out = backend.evolve()
    assert_sv_close(sv_out, math.sqrt(2)/2*StateVector([1, 0]) + math.sqrt(2)/2*StateVector([0, 1]))
    backend.set_input_state(BasicState([1, 1]))
    sv_out = backend.evolve()
    assert_sv_close(sv_out, math.sqrt(2)/2*StateVector([2, 0]) - math.sqrt(2)/2*StateVector([0, 2]))


def test_backend_mps_n_mode_perm_decomp():
    backend = BackendFactory.get_backend("MPS")
    backend.set_circuit(Circuit(3) // (0, PERM([2, 0, 1])) // (1, BS.H()))

    for r, c in backend._circuit:
        if isinstance(c, PERM):
            assert c.m == 2

    check_output_distribution(backend, BasicState("|2,0,0>"),
                              {BasicState("|0,2,0>"): 0.25,
                               BasicState("|0,1,1>"): 0.5,
                               BasicState("|0,0,2>"): 0.25})
    check_output_distribution(backend, BasicState("|1,0,0>"),
                              {BasicState("|0,1,0>"): 0.5,
                               BasicState("|0,0,1>"): 0.5})
    check_output_distribution(backend, BasicState("|1,0,1>"),
                              {BasicState("|0,2,0>"): 0.5,
                               BasicState("|0,0,2>"): 0.5})

@pytest.mark.parametrize("backend_name", ["SLOS", "Naive", "MPS"])
def test_probampli_iterator_cache(backend_name):
    b = BackendFactory.get_backend(backend_name)
    b.set_circuit(Circuit(5).add(0, BS.H()))
    b.set_input_state(BasicState([1, 1, 0, 0, 0]))
    b.evolve()
    assert len(b._cache_iterator) != 0
    b.set_circuit(Circuit(7).add(3, BS.H()))
    assert len(b._cache_iterator) == 0
