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
    from qat.lang.AQASM import Program, H, X, Y, Z, I, S, T, PH, RX, RY, RZ, CNOT, CSIGN, AbstractGate
except ModuleNotFoundError as e:
    pytest.skip("need `myqlm` module", allow_module_level=True)

from perceval import BasicState, StateVector, Circuit
from perceval.converters import MyQLMConverter
import perceval.components.unitary_components as comp
from perceval.components import catalog

# todo : implement further assertions


def test_basic_circuit_h():
    convertor = MyQLMConverter(catalog)
    qprog = Program()  # Create a Program
    qbits = qprog.qalloc(1)  # Allocate some qbits
    print(qbits, type(qbits))
    qprog.apply(H, qbits[0])  # Apply H gate
    myqlmc = qprog.to_circ()  # Export this program into a quantum circuit

    pc = convertor.convert(myqlmc)
    c = pc.linear_circuit()
    assert c.m == 2 * len(qbits)

    c0 = c._components[0][1]._components[0][1]
    assert isinstance(c0, comp.BS)


def test_cnot_1_heralded():
    convertor = MyQLMConverter(catalog)
    qprog = Program()
    qbits = qprog.qalloc(2)  # AllocateS 2 qbits
    qprog.apply(CNOT, qbits[0], qbits[1])
    myqlmc = qprog.to_circ()

    pc = convertor.convert(myqlmc, use_postselection=False)
    c = pc.linear_circuit()
    print('unitary')
    import perceval as pcvl
    pcvl.pdisplay(c.compute_unitary())

    gate_id = myqlmc.ops[0].gate
    gate_matrix = myqlmc.gateDic[gate_id].matrix  # gate matrix data from myQLM
    gate_u = MyQLMConverter._gate_def_nparray(gate_matrix)
    print(gate_u)

    assert pc.circuit_size == 8
    assert pc.m == 4


def test_cnot_H():
    convertor = MyQLMConverter(catalog)
    qprog = Program()
    qbits = qprog.qalloc(2)  # AllocateS 2 qbits
    qprog.apply(H, qbits[0])
    qprog.apply(CNOT, qbits[0], qbits[1])
    myqlmc = qprog.to_circ()

    pc = convertor.convert(myqlmc, use_postselection=False)
    assert pc.circuit_size == 8
    assert pc.m == 4


def test_cz_heralded():
    convertor = MyQLMConverter(catalog)
    qprog = Program()
    qbits = qprog.qalloc(2)  # AllocateS 2 qbits
    qprog.apply(CSIGN, qbits[0], qbits[1])  # CZ or Controlled Z gate is called CSIGN in myqlm
    myqlmc = qprog.to_circ()

    pc = convertor.convert(myqlmc)
    assert pc.circuit_size == 6  # 2 heralded modes for CZ
    assert pc.m == 4


def test_phase_shifter():
    convertor = MyQLMConverter(catalog)
    qprog = Program()
    qbits = qprog.qalloc(1)
    qprog.apply(PH(np.pi/2), qbits[0])  # PH -> phase shifter
    myqlmc = qprog.to_circ()

    pc = convertor.convert(myqlmc)
    assert pc.m == 2


@pytest.mark.parametrize('Gate_Name', [H, X, Y, Z, S, T, RX, RY, RZ])
def test_compare_u_1qbit(Gate_Name):
    # todo: I and PH not working
    qprog = Program()
    qbits = qprog.qalloc(1)

    if Gate_Name in ([RX, RY, RZ]):
        qprog.apply(Gate_Name(np.pi/2), qbits[0])
    else:
        qprog.apply(Gate_Name, qbits[0])

    circ = qprog.to_circ()
    gate_id = circ.ops[0].gate
    gate_matrix = circ.gateDic[gate_id].matrix  # gate matrix data from myQLM

    myqlm_converter = MyQLMConverter(catalog)
    myqlm_gate_u = myqlm_converter._myqlm_gate_unitary(gate_matrix)

    pcvl_proc = myqlm_converter.convert(circ, use_postselection=False)
    c = pcvl_proc.linear_circuit()
    cm = c.compute_unitary()
    diff = np.absolute(np.array(cm) - myqlm_gate_u)

    assert np.all(diff < 1e-5)  # todo: precision depends on optimize in converter


def test_abstract_1qbit_gate():

    # Generator method
    def Abs_Gate_generator(phi, theta):
        _I = np.eye(2, dtype=np.complex128)
        _X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        _Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
        return np.cos(theta / 2) * _I - 1j * np.sin(theta / 2) * (np.cos(phi) * _X + np.sin(phi) * _Y)

    # Some abstract gate with 2 parameters
    Abs_gate = AbstractGate("Abs_Gate", [float, float], arity=1)
    Abs_gate.set_matrix_generator(Abs_Gate_generator)

    prog = Program()
    qbits = prog.qalloc(1)
    prog.apply(Abs_gate(np.pi, np.pi / 3), qbits[0])  # simply testing with 2 arbitrary values
    circ = prog.to_circ()

    gate_id = circ.ops[0].gate
    gate_matrix = circ.gateDic[gate_id].matrix  # gate matrix data from myQLM

    myqlm_converter = MyQLMConverter(catalog)
    myqlm_gate_u = myqlm_converter._myqlm_gate_unitary(gate_matrix)

    pcvl_proc = myqlm_converter.convert(circ, use_postselection=False)
    c = pcvl_proc.linear_circuit()
    cm = c.compute_unitary()
    diff = np.absolute(np.array(cm) - myqlm_gate_u)

    assert np.all(diff < 1e-5)  # todo: precision depends on optimize in converter
    assert c.m == 2 * len(qbits)


def test_noon_state():
    # todo: implement a circuit using H and CNOT gates to prepare NOON states
    pass
