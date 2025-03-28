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
import numpy as np

try:
    from qat.lang.AQASM import Program, H, X, Y, Z, I, S, T, PH, RX, RY, RZ, SWAP, CNOT, CSIGN, AbstractGate
    from qat.core.circuit_builder.matrix_util import circ_to_np
except ModuleNotFoundError as e:
    assert e.name == "qat"
    pytest.skip("need `myqlm` module", allow_module_level=True)

from perceval import BasicState, StateVector
from perceval.converters import MyQLMConverter
import perceval.components.unitary_components as comp
from perceval.algorithm import Sampler


def test_basic_circuit_h():
    convertor = MyQLMConverter()
    qprog = Program()  # Create a Program
    qbits = qprog.qalloc(1)  # Allocate some qbits
    qprog.apply(H, qbits[0])  # Apply H gate
    myqlmc = qprog.to_circ()  # Export this program into a quantum circuit

    pc = convertor.convert(myqlmc)
    c = pc.linear_circuit()
    assert c.m == 2 * len(qbits)

    c0 = c._components[0][1]._components[0][1]
    assert isinstance(c0, comp.BS)

    sd = pc.source_distribution
    assert len(sd) == 1
    assert sd[StateVector('|1,0>')] == 1


def test_cnot_1_heralded():
    convertor = MyQLMConverter()
    qprog = Program()
    qbits = qprog.qalloc(2)  # AllocateS 2 qbits
    qprog.apply(CNOT, qbits[0], qbits[1])
    myqlmc = qprog.to_circ()

    pc = convertor.convert(myqlmc, use_postselection=False)
    assert pc.circuit_size == 6
    assert pc.m == 4
    assert pc.source_distribution[StateVector('|1,0,1,0,1,1>')] == 1


def test_cnot_H():
    convertor = MyQLMConverter()
    qprog = Program()
    qbits = qprog.qalloc(2)  # AllocateS 2 qbits
    qprog.apply(H, qbits[0])
    qprog.apply(CNOT, qbits[0], qbits[1])
    myqlmc = qprog.to_circ()

    pc = convertor.convert(myqlmc, use_postselection=False)
    assert pc.circuit_size == 6
    assert pc.m == 4
    assert pc.source_distribution[StateVector('|1,0,1,0,1,1>')] == 1


def test_cnot_1_postprocessed():
    convertor = MyQLMConverter()
    qprog = Program()
    qbits = qprog.qalloc(2)  # AllocateS 2 qbits
    qprog.apply(H, qbits[0])
    qprog.apply(CNOT, qbits[0], qbits[1])
    myqlmc = qprog.to_circ()

    pc = convertor.convert(myqlmc, use_postselection=True)
    assert pc.circuit_size == 6
    assert pc.source_distribution[StateVector('|1,0,1,0,0,0>')] == 1
    assert len(pc.components) == 2  # No permutation needed, only H and CNOT components exist in the Processor
    # should be BS//CNOT
    bsd_out = pc.probs()['results']
    assert len(bsd_out) == 2


def test_cz_heralded():
    convertor = MyQLMConverter()
    qprog = Program()
    qbits = qprog.qalloc(2)  # AllocateS 2 qbits
    qprog.apply(CSIGN, qbits[0], qbits[1])  # CZ or Controlled Z gate is called CSIGN in myqlm
    myqlmc = qprog.to_circ()

    pc = convertor.convert(myqlmc)
    assert pc.circuit_size == 6  # 2 heralded modes for CZ
    assert pc.m == 4
    bsd_out = pc.probs()['results']
    assert len(bsd_out) == 1


def test_basic_circuit_swap():
    convertor = MyQLMConverter()
    qprog = Program()
    qbits = qprog.qalloc(2)
    qprog.apply(SWAP, qbits[0], qbits[1])
    myqlmc = qprog.to_circ()

    pc = convertor.convert(myqlmc)
    assert pc.source_distribution[StateVector('|1,0,1,0>')] == 1
    assert len(pc.components) == 1
    r0, c0 = pc.components[0]
    assert r0 == (0, 1, 2, 3)
    assert isinstance(c0, comp.PERM)
    assert c0.perm_vector == [2, 3, 0, 1]


@pytest.mark.parametrize('Gate_Name', [H, PH, I, X, Y, Z, S, T, RX, RY, RZ])
def test_compare_u_1qbit(Gate_Name):
    qprog = Program()
    qbits = qprog.qalloc(1)

    if Gate_Name in ([PH, RX, RY, RZ]):
        qprog.apply(Gate_Name(math.pi/2), qbits[0])
    else:
        qprog.apply(Gate_Name, qbits[0])

    circ = qprog.to_circ()
    gate_id = circ.ops[0].gate
    gate_matrix = circ.gateDic[gate_id].matrix  # gate matrix data from myQLM

    myqlm_converter = MyQLMConverter()
    myqlm_gate_u = circ_to_np(gate_matrix)

    pcvl_proc = myqlm_converter.convert(circ, use_postselection=False)
    c = pcvl_proc.linear_circuit()
    cm = c.compute_unitary()
    diff = np.absolute(np.array(cm) - myqlm_gate_u)

    assert np.all(diff < 1e-4)  # precision depends on that of optimize in converter min_precision_gate set to 1e-4


def test_abstract_1qbit_gate():
    # Generator method
    def Abs_Gate_generator(phi, theta):
        _I = np.eye(2, dtype=np.complex128)
        _X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        _Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
        return math.cos(theta / 2) * _I - 1j * math.sin(theta / 2) * (math.cos(phi) * _X + math.sin(phi) * _Y)

    # Some abstract gate with 2 parameters
    Abs_gate = AbstractGate("Abs_Gate", [float, float], arity=1)
    Abs_gate.set_matrix_generator(Abs_Gate_generator)

    prog = Program()
    qbits = prog.qalloc(1)
    prog.apply(Abs_gate(math.pi, math.pi / 3), qbits[0])  # simply testing with 2 arbitrary values
    circ = prog.to_circ()

    gate_id = circ.ops[0].gate
    gate_matrix = circ.gateDic[gate_id].matrix  # gate matrix data from myQLM

    myqlm_converter = MyQLMConverter()
    myqlm_gate_u = circ_to_np(gate_matrix)

    pcvl_proc = myqlm_converter.convert(circ, use_postselection=False)
    c = pcvl_proc.linear_circuit()
    cm = c.compute_unitary()
    diff = np.absolute(np.array(cm) - myqlm_gate_u)

    assert np.all(diff < 1e-4)  # precision depends on that of optimize in converter min_precision_gate set to 1e-4
    assert c.m == 2 * len(qbits)


def test_converter_ghz_state():
    # output distribution being displayed to verify computation from converted circuit in perceval
    convertor = MyQLMConverter(backend_name="Naive")
    qprog = Program()
    qbits = qprog.qalloc(3)
    qprog.apply(H, qbits[0])
    qprog.apply(CNOT, qbits[0], qbits[1])
    qprog.apply(CNOT, qbits[0], qbits[2])
    myqlmc = qprog.to_circ()

    pc = convertor.convert(myqlmc, use_postselection=True)
    assert pc.m == 6
    assert pc.circuit_size == 10  # m + heralded modes = m + nb_cnot*2
    import perceval as pcvl
    pc.with_input(pcvl.LogicalState([0, 0, 0]))
    sampler = Sampler(pc)
    output_distribution = sampler.probs()["results"]
    # GHZ state distribution is expected ( # precision is a bit off because of heralded CNOT implementation)
    logical000 = BasicState('|1,0,1,0,1,0>')
    logical111 = BasicState('|0,1,0,1,0,1>')
    assert output_distribution[logical000] == pytest.approx(0.5, abs=1e-3)
    assert output_distribution[logical111] == pytest.approx(0.5, abs=1e-3)


@pytest.mark.skip(reason="Only for Dev, takes long for computation and displays truth table")
def test_converter_noon_state():
    convertor = MyQLMConverter(backend_name="SLOS")
    qprog = Program()
    qbits = qprog.qalloc(4)
    qprog.apply(H, qbits[0])
    qprog.apply(CNOT, qbits[0], qbits[1])
    qprog.apply(CNOT, qbits[0], qbits[2])
    qprog.apply(CNOT, qbits[0], qbits[3])
    myqlmc = qprog.to_circ()

    pc = convertor.convert(myqlmc, use_postselection=True)
    import perceval as pcvl
    pcvl.pdisplay(pc)
    pc.with_input(pcvl.LogicalState([0, 0, 0, 0]))

    sampler = Sampler(pc)
    assert pc.m == 2 * len(qbits)
    output_distribution = sampler.probs()["results"]
    pcvl.pdisplay(output_distribution, precision=1e-2, max_v=4)


def test_circuit_measure():
    prog = Program()  # AQASM program in myqlm
    qbits = prog.qalloc(2)
    for qb in qbits:
        H(qb)
    CNOT(qbits[0], qbits[1])
    prog.measure(qbits[1])

    circuit = prog.to_circ()
    converter = MyQLMConverter()
    with pytest.raises(AssertionError):
        converter.convert(circuit, use_postselection=True)


def test_circuit_parametrized_gates():
    prog = Program()
    qbits = prog.qalloc(4)
    PH(1.22)(qbits[0])
    RY(3.14)(qbits[1])
    RZ(3.14)(qbits[2])
    RX(-3.14)(qbits[3])
    circuit = prog.to_circ()

    converter = MyQLMConverter()
    pc = converter.convert(circuit, use_postselection=True)

    assert len(pc.components) == 4
    assert isinstance(pc.components[0][1]._components[0][1], comp.PS)
    assert isinstance(pc.components[1][1]._components[0][1], comp.BS)

    assert isinstance(pc.components[2][1]._components[0][1], comp.PS)
    assert isinstance(pc.components[2][1]._components[0][1], comp.PS)

    assert isinstance(pc.components[3][1]._components[0][1], comp.BS)
