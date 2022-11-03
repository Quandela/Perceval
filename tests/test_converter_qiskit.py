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

try:
    import qiskit
except ModuleNotFoundError as e:
    pytest.skip("need `qiskit` module", allow_module_level=True)

from perceval import BasicState, StateVector, Circuit
from perceval.converters import QiskitConverter
import perceval.components.unitary_components as comp
from perceval.components import catalog


def test_basic_circuit_h():
    convertor = QiskitConverter(catalog)
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
    convertor = QiskitConverter(catalog)
    qc = qiskit.QuantumCircuit(1)
    qc.h(0)
    qc.h(0)
    pc = convertor.convert(qc)
    assert pc.source_distribution[StateVector('|1,0>')] == 1
    assert len(pc._components) == 2


def test_basic_circuit_s():
    convertor = QiskitConverter(catalog)
    qc = qiskit.QuantumCircuit(1)
    qc.s(0)
    pc = convertor.convert(qc)
    assert pc.source_distribution[StateVector('|1,0>')] == 1
    assert len(pc._components) == 1
    assert isinstance(pc._components[0][1], Circuit) and len(pc._components[0][1]._components) == 1
    r0 = pc._components[0][1]._components[0][0]
    c0 = pc._components[0][1]._components[0][1]
    assert r0 == (1,)
    assert isinstance(c0, comp.PS)


def test_basic_circuit_swap_direct():
    convertor = QiskitConverter(catalog)
    qc = qiskit.QuantumCircuit(2)
    qc.swap(0, 1)
    pc = convertor.convert(qc)
    assert pc.source_distribution[StateVector('|1,0,1,0>')] == 1
    assert len(pc._components) == 1
    r0, c0 = pc._components[0]
    assert r0 == [0, 1, 2, 3]
    assert isinstance(c0, comp.PERM)
    assert c0.perm_vector == [2, 3, 0, 1]


def test_basic_circuit_swap_indirect():
    convertor = QiskitConverter(catalog)
    qc = qiskit.QuantumCircuit(2)
    qc.swap(1, 0)
    pc = convertor.convert(qc)
    assert pc.source_distribution[StateVector('|1,0,1,0>')] == 1
    assert len(pc._components) == 1
    r0, c0 = pc._components[0]
    assert r0 == [0, 1, 2, 3]
    assert isinstance(c0, comp.PERM)
    assert c0.perm_vector == [2, 3, 0, 1]


def test_cnot_1_heralded():
    convertor = QiskitConverter(catalog)
    qc = qiskit.QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    pc = convertor.convert(qc, use_postselection=False)
    assert pc.circuit_size == 8
    assert pc.m == 4
    assert pc.source_distribution[StateVector('|1,0,1,0,0,1,0,1>')] == 1
    assert len(pc._components) == 4
    # should be BS//PERM//CNOT//PERM
    perm1 = pc._components[1][1]
    assert isinstance(perm1, comp.PERM)
    perm2 = pc._components[3][1]
    assert isinstance(perm2, comp.PERM)
    # check that ports are correctly connected
    assert perm1.perm_vector == [2, 3, 4, 5, 0, 1, 6, 7]
    assert perm2.perm_vector == [4, 5, 0, 1, 2, 3, 6, 7]


def test_cnot_1_inverse_heralded():
    convertor = QiskitConverter(catalog)
    qc = qiskit.QuantumCircuit(2)
    qc.h(0)
    qc.cx(1, 0)
    pc = convertor.convert(qc, use_postselection=False)
    assert pc.circuit_size == 8
    assert pc.m == 4
    assert pc.source_distribution[StateVector('|1,0,1,0,0,1,0,1>')] == 1
    assert len(pc._components) == 4
    # should be BS//PERM//CNOT//PERM
    perm1 = pc._components[1][1]
    assert isinstance(perm1, comp.PERM)
    perm2 = pc._components[3][1]
    assert isinstance(perm2, comp.PERM)
    # check that ports are correctly connected
    assert perm1.perm_vector == [4, 5, 2, 3, 0, 1, 6, 7]
    assert perm2.perm_vector == [4, 5, 2, 3, 0, 1, 6, 7]


def test_cnot_2_heralded():
    convertor = QiskitConverter(catalog)
    qc = qiskit.QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 2)
    pc = convertor.convert(qc, use_postselection=False)
    assert pc.circuit_size == 10
    assert pc.m == 6
    assert pc.source_distribution[StateVector('|1,0,1,0,1,0,0,1,0,1>')] == 1
    assert len(pc._components) == 4
    # should be BS//PERM//CNOT//PERM
    perm1 = pc._components[1][1]
    assert isinstance(perm1, comp.PERM)
    perm2 = pc._components[3][1]
    assert isinstance(perm2, comp.PERM)
    # check that ports are correctly connected
    assert perm1.perm_vector == [2, 3, 8, 9, 4, 5, 0, 1, 6, 7]
    assert perm2.perm_vector == [6, 7, 0, 1, 4, 5, 8, 9, 2, 3]


def test_cnot_1_postprocessed():
    convertor = QiskitConverter(catalog)
    qc = qiskit.QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    pc = convertor.convert(qc, use_postselection=True)
    assert pc.circuit_size == 6
    assert pc.source_distribution[StateVector('|1,0,1,0,0,0>')] == 1
    assert len(pc._components) == 4
    # should be BS//PERM//CNOT//PERM
    perm1 = pc._components[1][1]
    assert isinstance(perm1, comp.PERM)
    perm2 = pc._components[3][1]
    assert isinstance(perm2, comp.PERM)
    # check that ports are correctly connected
    assert perm1.perm_vector == [1, 2, 3, 4, 0, 5]
    assert perm2.perm_vector == [4, 0, 1, 2, 3, 5]


def test_cnot_postprocess():
    convertor = QiskitConverter(catalog)
    qc = qiskit.QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    pc = convertor.convert(qc)
    bsd_out = pc.probs()['results']
    assert len(bsd_out) == 2


def test_cnot_herald():
    convertor = QiskitConverter(catalog)
    qc = qiskit.QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    pc = convertor.convert(qc, True)
    bsd_out = pc.probs()['results']
    assert bsd_out[BasicState("|1,0,0,1>")] + bsd_out[BasicState("|0,1,1,0>")] < 2e-5
    assert bsd_out[BasicState("|1,0,1,0>")] + bsd_out[BasicState("|0,1,0,1>")] > 0.99
    assert len(bsd_out) == 4
