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
import numpy as np

try:
    import qiskit
except ModuleNotFoundError as e:
    assert e.name == "qiskit"
    pytest.skip("need `qiskit` module", allow_module_level=True)

from perceval import BasicState, StateVector, Circuit
from perceval.converters import QiskitConverter
from perceval.converters.qiskit_converter import _get_gate_sequence
from perceval.converters.converter_utils import label_cnots_in_gate_sequence
import perceval.components.unitary_components as comp
from perceval.components.port import get_basic_state_from_encoding
from perceval.utils import BSDistribution, Encoding, generate_all_logical_states


EXPECTED_QISK_SIM_PROBS_DATA = {
    "0000": 0.18158257565856697,
    "1000": 0.15748294604956967,
    "0100": 0.02913976729404776,
    "1100": 0.025272338956653793,
    "0010": 0.07928849942312567,
    "1010": 0.06876533407303459,
    "0110": 0.05447209941994839,
    "1110": 0.04724256533451039,
    "0001": 0.10070839860443088,
    "1001": 0.08734238539485045,
    "0101": 0.01616134857238361,
    "1101": 0.014016415264968182,
    "0011": 0.04397458167828672,
    "1011": 0.03813827757909818,
    "0111": 0.030211036941779033,
    "1111": 0.026201429754745716
}


def test_basic_circuit_h():
    convertor = QiskitConverter()
    qc = qiskit.QuantumCircuit(1)
    qc.h(0)
    pc = convertor.convert(qc)
    c = pc.linear_circuit()
    sd = pc.source_distribution
    assert len(sd) == 1
    assert sd[StateVector('|1,0>')] == 1
    assert len(c._components) == 1
    assert isinstance(c._components[0][1], Circuit) and len(c._components[0][1]._components) == 1
    c0 = c._components[0][1]._components[0][1]
    assert isinstance(c0, comp.BS)
    assert c0._convention == comp.BSConvention.H


def test_basic_circuit_double_h():
    convertor = QiskitConverter()
    qc = qiskit.QuantumCircuit(1)
    qc.h(0)
    qc.h(0)
    pc = convertor.convert(qc)
    assert pc.source_distribution[StateVector('|1,0>')] == 1
    assert len(pc._components) == 2


def test_basic_circuit_s():
    convertor = QiskitConverter()
    qc = qiskit.QuantumCircuit(1)
    qc.s(0)
    pc = convertor.convert(qc)
    assert pc.source_distribution[StateVector('|1,0>')] == 1
    assert len(pc.components) == 1
    assert isinstance(pc.components[0][1], Circuit) and len(pc.components[0][1]._components) == 1
    r0 = pc.components[0][1]._components[0][0]
    c0 = pc.components[0][1]._components[0][1]
    assert r0 == (1,)
    assert isinstance(c0, comp.PS)


def test_basic_circuit_swap_direct():
    convertor = QiskitConverter()
    qc = qiskit.QuantumCircuit(2)
    qc.swap(0, 1)
    pc = convertor.convert(qc)
    assert pc.source_distribution[StateVector('|1,0,1,0>')] == 1
    assert len(pc.components) == 1
    r0, c0 = pc.components[0]
    assert r0 == (0, 1, 2, 3)
    assert isinstance(c0, comp.PERM)
    assert c0.perm_vector == [2, 3, 0, 1]


def test_basic_circuit_swap_indirect():
    convertor = QiskitConverter()
    qc = qiskit.QuantumCircuit(2)
    qc.swap(1, 0)
    pc = convertor.convert(qc)
    assert pc.source_distribution[StateVector('|1,0,1,0>')] == 1
    assert len(pc.components) == 1
    r0, c0 = pc.components[0]
    assert r0 == (0, 1, 2, 3)
    assert isinstance(c0, comp.PERM)
    assert c0.perm_vector == [2, 3, 0, 1]


def test_basic_circuit_swap_with_gap():
    convertor = QiskitConverter()
    qc = qiskit.QuantumCircuit(5)
    qc.swap(1, 3)
    pc = convertor.convert(qc)
    assert len(pc.components) == 1
    r0, c0 = pc.components[0]
    assert r0 == (2, 3, 4, 5, 6, 7)
    assert isinstance(c0, comp.PERM)
    assert c0.perm_vector == [4, 5, 2, 3, 0, 1]


def test_cnot_1_heralded():
    convertor = QiskitConverter()
    qc = qiskit.QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    pc = convertor.convert(qc, use_postselection=False)
    assert pc.circuit_size == 6
    assert pc.m == 4
    assert pc.source_distribution[StateVector('|1,0,1,0,1,1>')] == 1
    assert len(pc.components) == 2  # should be BS.H//CNOT


def test_cnot_1_inverse_heralded():
    convertor = QiskitConverter()
    qc = qiskit.QuantumCircuit(2)
    qc.h(0)
    qc.cx(1, 0)
    pc = convertor.convert(qc, use_postselection=False)
    assert pc.circuit_size == 6
    assert pc.m == 4
    assert pc.source_distribution[StateVector('|1,0,1,0,1,1>')] == 1
    assert len(pc.components) == 4
    # should be BS//PERM//CNOT//PERM
    perm1 = pc.components[1][1]
    assert isinstance(perm1, comp.PERM)
    perm2 = pc.components[3][1]
    assert isinstance(perm2, comp.PERM)
    # check that ports are correctly connected
    assert perm1.perm_vector == [2, 3, 0, 1]
    assert perm2.perm_vector == [2, 3, 0, 1]


def test_cnot_2_heralded():
    convertor = QiskitConverter()
    qc = qiskit.QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 2)
    pc = convertor.convert(qc, use_postselection=False)
    assert pc.circuit_size == 8
    assert pc.m == 6
    assert pc.source_distribution[StateVector('|1,0,1,0,1,0,1,1>')] == 1
    assert len(pc.components) == 4
    # should be BS//PERM//CNOT//PERM
    perm1 = pc.components[1][1]
    assert isinstance(perm1, comp.PERM)
    perm2 = pc.components[3][1]
    assert isinstance(perm2, comp.PERM)
    # check that ports are correctly connected
    assert perm1.perm_vector == [4, 5, 0, 1, 2, 3]
    assert perm2.perm_vector == [2, 3, 4, 5, 0, 1]


def test_cnot_1_postprocessed():
    convertor = QiskitConverter()
    qc = qiskit.QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    pc = convertor.convert(qc, use_postselection=True)
    assert pc.circuit_size == 6
    assert pc.source_distribution[StateVector('|1,0,1,0,0,0>')] == 1
    assert len(pc.components) == 2  # No permutation needed, only H and CNOT components exist in the Processor
    # should be BS//CNOT


def test_cnot_postprocess():
    convertor = QiskitConverter()
    qc = qiskit.QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    pc = convertor.convert(qc, use_postselection=True)
    bsd_out = pc.probs()['results']
    assert len(bsd_out) == 2

    qc.h(0)  # We should be able to continue the circuit with 1-qubit gates even with a post-selected CNOT
    pc = convertor.convert(qc, use_postselection=True)
    assert isinstance(pc.components[-1][1]._components[0][1], comp.BS)


def test_cnot_herald():
    convertor = QiskitConverter()
    qc = qiskit.QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    pc = convertor.convert(qc, True)
    bsd_out = pc.probs()['results']
    assert bsd_out[BasicState("|1,0,0,1>")] + bsd_out[BasicState("|0,1,1,0>")] < 2e-5
    assert bsd_out[BasicState("|1,0,1,0>")] + bsd_out[BasicState("|0,1,0,1>")] > 0.99
    assert len(bsd_out) == 4

def qiskit_circ_multiple_cnots():
    # Gate Circuit
    params = [0.55254396, 0.97997692, 0.17090731,
              0.33458378, 0.94711577, 0.39304346, 0.98720513, 0.94564894]

    circ = qiskit.QuantumCircuit(4)

    circ.rx(params[0], 0)
    circ.ry(params[1], 1)
    circ.rx(params[2], 2)
    circ.rx(params[3], 3)

    circ.cx(1, 2)

    circ.rx(params[4], 0)
    circ.ry(params[5], 1)
    circ.rx(params[6], 2)
    circ.rx(params[7], 3)

    circ.cx(0, 1)
    circ.cx(0, 1)
    circ.cx(2, 3)
    circ.cx(2, 3)
    return circ

def test_cnot_ppcnot_vs_hcnot():
    qisk_circ = qiskit_circ_multiple_cnots()

    converter = QiskitConverter()
    pc = converter.convert(qisk_circ, use_postselection=True)  # converted

    # Verify correct number and position of post-processed CNOTs in conversion
    gate_seq_converted = []
    for _, c in pc.components:
        gate_seq_converted.append(c.name)

    gate_seq_qisk = _get_gate_sequence(qisk_circ) # gate list from qiskit
    optimized_gate_sequence = label_cnots_in_gate_sequence(gate_seq_qisk)

    num_ppcnot_expt = len([elem for elem in optimized_gate_sequence if elem == 'postprocessed cnot'])
    cnot_order_expct = [elem.lower() for elem in optimized_gate_sequence if elem.endswith('cnot')]

    cnot_order_converted = [elem.lower() for elem in gate_seq_converted if elem.endswith('CNOT')]
    num_ppcnot_pc = len([elem for elem in gate_seq_converted if elem == 'PostProcessed CNOT'])

    assert cnot_order_converted == cnot_order_expct
    assert num_ppcnot_pc == num_ppcnot_expt  # compare number of Ralphs


def test_cnot_ppcnot_vs_hcnot_sim():
    qisk_circ = qiskit_circ_multiple_cnots()

    converter = QiskitConverter()
    pc = converter.convert(qisk_circ, use_postselection=True)  # converted

    probs_pc = pc.probs()['results']

    # convert the probs to valid logical states
    num_qubits = 4
    enc_list = [Encoding.DUAL_RAIL]*4
    logical_states = generate_all_logical_states(num_qubits)
    basic_states = []
    for each_ls in logical_states:
        basic_states.append(get_basic_state_from_encoding(enc_list, each_ls))

    # extracting logically valid samples from perceval output
    valid_bsd = BSDistribution()
    for each_bs in basic_states:
        if each_bs in list(probs_pc.keys()):
            valid_bsd.add(each_bs, probs_pc[each_bs])
    valid_bsd.normalize()

    tot_diff = 0  # sum of absolute difference between the probs of the two computations
    for index, (key, value) in enumerate(valid_bsd.items()):
        bit_form = np.binary_repr(index, 4)  # bit form of basic states - to extract data from Qiskit output
        tot_diff += abs(value - EXPECTED_QISK_SIM_PROBS_DATA[bit_form])

    assert tot_diff == pytest.approx(0, abs=1e-7)
