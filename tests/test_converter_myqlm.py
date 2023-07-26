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
    from qat.qpus import PyLinalg
except ModuleNotFoundError as e:
    pytest.skip("need `myqlm` module", allow_module_level=True)

from perceval import BasicState, StateVector, Circuit
from perceval.converters import MyQLMConverter
import perceval.components.unitary_components as comp
from perceval.components import catalog
from perceval.algorithm import Analyzer, Sampler
from perceval.utils.stategenerator import StateGenerator
from perceval.utils import Encoding


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

    sd = pc.source_distribution
    assert len(sd) == 1
    assert sd[StateVector('|1,0>')] == 1


def test_cnot_1_heralded():
    convertor = MyQLMConverter(catalog)
    qprog = Program()
    qbits = qprog.qalloc(2)  # AllocateS 2 qbits
    qprog.apply(CNOT, qbits[0], qbits[1])
    myqlmc = qprog.to_circ()

    pc = convertor.convert(myqlmc, use_postselection=False)
    c = pc.linear_circuit()

    assert pc.circuit_size == 8
    assert pc.m == 4
    assert pc.source_distribution[StateVector('|1,0,1,0,0,1,0,1>')] == 1


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
    assert pc.source_distribution[StateVector('|1,0,1,0,0,1,0,1>')] == 1


def test_cnot_1_postprocessed():
    convertor = MyQLMConverter(catalog)
    qprog = Program()
    qbits = qprog.qalloc(2)  # AllocateS 2 qbits
    qprog.apply(H, qbits[0])
    qprog.apply(CNOT, qbits[0], qbits[1])
    myqlmc = qprog.to_circ()

    pc = convertor.convert(myqlmc, use_postselection=True)
    assert pc.circuit_size == 6
    assert pc.source_distribution[StateVector('|1,0,1,0,0,0>')] == 1
    assert len(pc._components) == 2  # No permutation needed, only H and CNOT components exist in the Processor
    # should be BS//CNOT
    bsd_out = pc.probs()['results']
    assert len(bsd_out) == 2


def test_cz_heralded():
    convertor = MyQLMConverter(catalog)
    qprog = Program()
    qbits = qprog.qalloc(2)  # AllocateS 2 qbits
    qprog.apply(CSIGN, qbits[0], qbits[1])  # CZ or Controlled Z gate is called CSIGN in myqlm
    myqlmc = qprog.to_circ()

    pc = convertor.convert(myqlmc)
    assert pc.circuit_size == 6  # 2 heralded modes for CZ
    assert pc.m == 4
    bsd_out = pc.probs()['results']
    assert len(bsd_out) == 1


def test_phase_shifter():
    # todo: fix U, mode verification
    convertor = MyQLMConverter(catalog)
    qprog = Program()
    qbits = qprog.qalloc(1)
    qprog.apply(PH(np.pi/3), qbits[0])  # PH -> phase shifter
    myqlmc = qprog.to_circ()

    pc = convertor.convert(myqlmc)
    assert pc.m == 2
    gate_id = myqlmc.ops[0].gate
    gate_matrix = myqlmc.gateDic[gate_id].matrix  # gate matrix data from myQLM

    myqlm_converter = MyQLMConverter(catalog)
    myqlm_gate_u = myqlm_converter._myqlm_gate_unitary(gate_matrix)

    pcvl_proc = myqlm_converter.convert(myqlmc, use_postselection=False)
    c = pcvl_proc.linear_circuit()
    cm = c.compute_unitary()
    diff = np.absolute(np.array(cm) - myqlm_gate_u)
    print('myqlm \n', myqlm_gate_u)
    print('pcvl \n', cm)
    print("diff \n", diff)


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

    assert np.all(diff < 1e-4)  # precision depends on that of optimize in converter min_precision_gate set to 1e-4


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

    assert np.all(diff < 1e-4)  # precision depends on that of optimize in converter min_precision_gate set to 1e-4
    assert c.m == 2 * len(qbits)


def test_converter_ghz_state():
    # todo : comparison of state to logical in results - remove display and add assertions
    convertor = MyQLMConverter(catalog, backend_name="Naive")
    qprog = Program()
    qbits = qprog.qalloc(3)
    qprog.apply(H, qbits[0])
    qprog.apply(CNOT, qbits[0], qbits[1])
    qprog.apply(CNOT, qbits[0], qbits[2])
    myqlmc = qprog.to_circ()

    pc = convertor.convert(myqlmc, use_postselection=True)
    assert pc.m == 6
    assert pc.circuit_size == 12 # m + heralded modes
    import perceval as pcvl
    pc.with_input(pcvl.LogicalState([0, 0, 0]))
    sampler = Sampler(pc)
    output_distribution = sampler.probs()["results"]
    assert sum(list(output_distribution.values())) == 1


    pcvl.pdisplay(output_distribution)

    # trying to see if i can compare actual results
    print(output_distribution)
    # Create a job
    pylinalgqpu = PyLinalg()
    job = myqlmc.to_job()
    result = pylinalgqpu.submit(job)  # Submit the job to the QPU

    # Iterate over the final state vector to get all final components
    for sample in result:
        print("State %s amplitude %s" % (sample.state, sample.amplitude))

        myqlm_logical_state_list = (list(sample.state.bitstring))  # obtained in string format
        logical_state = [int(i) for i in myqlm_logical_state_list]  # pcvl needs int/float
        print("myqlm output state in list", (logical_state))
        sg = StateGenerator(encoding=Encoding.DUAL_RAIL)
        sv_logical = sg.logical_state(logical_state)  # statevector
        print("myqlm list turned to logical state", sv_logical)


def test_converter_noon_state():
    # todo: decide if we need it, ghz confirms they are working - this is slower even with SLOS
    convertor = MyQLMConverter(catalog, backend_name="SLOS")
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

    output_distribution = sampler.probs()["results"]
    pcvl.pdisplay(output_distribution, precision=1e-2, max_v=4)
