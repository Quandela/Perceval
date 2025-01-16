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

try:
    import cqasm
except ModuleNotFoundError as e:
    assert e.name == "cqasm"
    pytest.skip("need `cqasm` module", allow_module_level=True)

import numpy as np
from pathlib import Path

from perceval.components import BS
from perceval.converters import CQASMConverter, ConversionSyntaxError, ConversionUnsupportedFeatureError, ConversionBadVersionError
from perceval.utils import BasicState, StateVector


def test_converter_version_check():
    assert CQASMConverter.check_version("// Comment\n\nversion 3\nq") == (3, 0)
    assert CQASMConverter.check_version("version 1.0\nqubits 2") == (1, 0)
    assert CQASMConverter.check_version(" version 2.5 ") == (2, 5)
    with pytest.raises(ConversionSyntaxError):
        assert CQASMConverter.check_version("version Z.X\n")
    with pytest.raises(ConversionSyntaxError):
        assert CQASMConverter.check_version("inversion 1.0\n")
    with pytest.raises(ConversionSyntaxError):
        assert CQASMConverter.check_version("//Comment \nvresion 4.0")


def test_converter_bad_version():
    v7_program = """version 7\nqubit q\n"""
    with pytest.raises(ConversionBadVersionError):
        CQASMConverter().convert(v7_program)


def test_converter_syntax_error():
    cqasm_program = """
version 3
rabbit r
"""
    with pytest.raises(ConversionSyntaxError):
        CQASMConverter().convert(cqasm_program)


def test_converter_unsupported_classical_variable():
    cqasm_program = """
version 3
qubit[2] q
bit b
X q[0]
"""
    with pytest.raises(ConversionUnsupportedFeatureError):
        CQASMConverter().convert(cqasm_program)


def test_converter_unsupported_gates():
    cqasm_program = """
version 3
qubit[3] q
CNOT q[0:2], q[2]
"""
    # Caught early: two controls for one target
    with pytest.raises(ConversionUnsupportedFeatureError):
        CQASMConverter().convert(cqasm_program)

    cqasm_program = """
version 3
qubit[3] q
CR(pi / 2) q[1], q[0]
"""
    with pytest.raises(ConversionUnsupportedFeatureError):
        CQASMConverter().convert(cqasm_program)


def test_converter_bell_state():
    cqasm_program = """
version 3
qubit[2] q
H q[0]
CNOT q[0], q[1]
"""
    pc = CQASMConverter().convert(cqasm_program, use_postselection=False)
    assert pc.circuit_size == 6
    assert pc.m == 4
    assert pc.source_distribution[StateVector('|1,0,1,0,1,1>')] == 1
    assert len(pc._components) == 2
    assert pc.components[0][1].name == "H"
    assert pc.components[1][1].name == "Heralded CNOT"
    r = pc.probs()['results']
    assert np.allclose(r[BasicState("|0, 1, 0, 1>")], 0.5)
    assert np.allclose(r[BasicState("|1, 0, 1, 0>")], 0.5)


def test_converter_bell_state_swapped():
    cqasm_program = """
version 3
qubit[2] q
H q[0]
CNOT q[1], q[0]
"""
    pc = CQASMConverter().convert(cqasm_program, use_postselection=False)
    assert pc.circuit_size == 6
    assert pc.m == 4
    assert pc.source_distribution[StateVector('|1,0,1,0,1,1>')] == 1
    assert len(pc._components) == 4  # should be  BS.H // PERM // CNOT // PERM
    assert pc.components[0][1].name == "H"
    assert pc.components[1][1].name == "PERM"
    assert pc.components[2][1].name == "Heralded CNOT"
    assert pc.components[3][1].name == "PERM"


def test_converter_postselection():
    cqasm_program = """
version 3
qubit[2] q
H q[0]
CNOT q[0], q[1]
"""
    pc = CQASMConverter().convert(cqasm_program, use_postselection=True)
    bsd_out = pc.probs()['results']
    assert pc.circuit_size == 6
    assert pc.source_distribution[StateVector('|1,0,1,0,0,0>')] == 1
    assert len(pc._components) == 2
    assert pc.components[0][1].name == "H"
    assert pc.components[1][1].name == "PostProcessed CNOT"
    assert len(bsd_out) == 2

    cqasm_program = """
version 3
qubit[2] q
H q[0]
CNOT q[0], q[1]
H q[0]
"""
    pc = CQASMConverter().convert(cqasm_program, use_postselection=True)
    assert isinstance(pc._components[-1][1]._components[0][1], BS)


def test_converter_qubit_names():
    cqasm_program = """
version 3
qubit alice
qubit bob
qubit[2] psi
X alice
Y bob
Z psi[0:1]
"""
    pc = CQASMConverter().convert(cqasm_program)
    assert tuple(pc.in_port_names) == \
        ("alice", "alice", "bob", "bob", "psi[0]", "psi[0]", "psi[1]", "psi[1]")


def test_converter_multi_target_cnot():
    cqasm_program = """
version 3
qubit[3] q
CNOT q[0], q[1:2]
"""
    pc = CQASMConverter().convert(cqasm_program, use_postselection=False)
    # Two heralded CNOTs sandwiched between PERMs
    assert len(pc.components) == 5
    assert pc.components[0][1].name == "PERM"
    assert pc.components[1][1].name == "Heralded CNOT"
    assert pc.components[2][1].name == "PERM"
    assert pc.components[3][1].name == "Heralded CNOT"
    assert pc.components[4][1].name == "PERM"


"""
References:
    - cQASM v3 for the list of available gates. https://qutech-delft.github.io/cQASM-spec/language_specification/instructions/gates.html#standard-gate-set
    - Qiskit to generate the expected unitaries
"""
_invsqrt2 = 1.0 / np.sqrt(2)
_eipi4 = np.exp(1j * np.pi / 4)
expected_unitaries = [
    ("H", _invsqrt2 * np.array([[1, 1], [1, -1]])),
    ("X", np.array([[0, 1], [1, 0]])),
    ("X90", _invsqrt2 * np.array([[1, -1j], [-1j, 1]])),
    ("mX90", _invsqrt2 * np.array([[1, 1j], [1j, 1]])),
    ("Rx(pi / 2)", _invsqrt2 * np.array([[1, -1j], [-1j, 1]])),
    ("Y", np.array([[0, -1j], [1j, 0]])),
    ("Y90", _invsqrt2 * np.array([[1, -1], [1, 1]])),
    ("mY90", _invsqrt2 * np.array([[1, 1], [-1, 1]])),
    ("Ry(pi / 2)", _invsqrt2 * np.array([[1, -1], [1, 1]])),
    ("Z", np.diag([1, -1])),
    ("S", np.diag([1, 1j])),
    ("Sdag", np.diag([1, -1j])),
    ("T", np.diag([1, _eipi4])),
    ("Tdag", np.diag([1, _eipi4.conj()])),
    ("Rz(pi / 2)", np.diag([_eipi4.conj(), _eipi4])),
]


@pytest.mark.parametrize(
    "gate_name,expected_unitary",
    expected_unitaries
)
def test_converter_one_qubit_gate(gate_name, expected_unitary):
    cqasm_program_template = f"""
version 3
qubit q
{ gate_name } q
"""
    pc = CQASMConverter().convert(cqasm_program_template, use_postselection=False)
    modes, circuit = pc.components[0]
    assert tuple(modes) == (0, 1)

    u = circuit.compute_unitary()
    assert np.allclose(u, expected_unitary)


def test_converter_from_file():
    TEST_DATA_DIR = Path(__file__).resolve().parent / 'data'
    cqasm_program_file = TEST_DATA_DIR / 'state_preparation_5.cqasm3'
    pc = CQASMConverter().convert(str(cqasm_program_file), use_postselection=False)
    assert pc.circuit_size == 10
    assert len(pc.heralds) == 4
    assert pc.m == 6
    assert len(pc._components) == 12
    r = pc.probs()['results']
    assert np.allclose(r[BasicState("|1, 0, 1, 0, 1, 0>")], 0.2, atol=0.01)
    assert np.allclose(r[BasicState("|1, 0, 1, 0, 0, 1>")], 0.2, atol=0.01)
    assert np.allclose(r[BasicState("|1, 0, 0, 1, 1, 0>")], 0.2, atol=0.01)
    assert np.allclose(r[BasicState("|1, 0, 0, 1, 0, 1>")], 0.2, atol=0.01)
    assert np.allclose(r[BasicState("|0, 1, 1, 0, 1, 0>")], 0.2, atol=0.01)


def test_converter_from_ast():
    converter = CQASMConverter()
    ast = converter._cqasm.Analyzer().analyze_string("version 3\nqubit q\nH q")
    pc = converter.convert(ast)
    assert pc.circuit_size == 2
    assert len(pc._components) == 1


# Tests for v1 converter
def test_converter_v1():
    cqasm_program = f"""
version 1.0

# a basic cQASM example
qubits 2

.prepare
    prep_z q[0:1]

.entangle
    H q[0]
    CNOT q[0], q[1]

.measurement
    measure_all
"""
    pc = CQASMConverter().convert(cqasm_program, use_postselection=False)
    assert pc.circuit_size == 6
    assert pc.m == 4
    assert len(pc._components) == 2
    assert pc.components[0][1].name == "H"
    assert pc.components[1][1].name == "Heralded CNOT"


def test_converter_v1_qubits_ge_10():
    source = f"""
version 1.0

# a basic cQASM example
qubits 11

.prepare
    prep_z q[0:1]

.entangle
    H q[0]
    CNOT q[10], q[6]
    Z q[5]

.measurement
    measure_all
"""
    pc = CQASMConverter().convert(source, use_postselection=False)
    assert pc.circuit_size == 24
    assert pc.m == 22
    assert len(pc._components) == 5


def test_converter_v1_controls_g2():
    source = f"""
version 1.0
# a simple cQASM v1 file

qubits 3

.entangle
    H q[0]
    Toffoli q[0], q[1], q[3]

.measurement
    measure_all
"""
    with pytest.raises(ConversionUnsupportedFeatureError):
        CQASMConverter().convert(source)


def test_converter_v1_syntax_error():
    source = f"""
version 1.0

qubits 3

.entangle
    hjcd q[0]
"""
    with pytest.raises(ConversionUnsupportedFeatureError):
        CQASMConverter().convert(source)

def test_converter_v1_qubits_error():
    source = f"""
version 1.0

qubits w

.entangle
    H q[0]
"""
    with pytest.raises(ConversionSyntaxError):
        CQASMConverter().convert(source)


def test_converter_v1_rotation_gates():
    source = f"""
version 1.0

qubits 2

.rotation
    Rx q[0], 3.14
"""
    pc = CQASMConverter().convert(source, use_postselection=False)
    assert pc.circuit_size == 4
    assert pc.source_distribution[StateVector('|1,0,1,0>')] == 1

def test_converter_multi_target_1qubit():
    # cqasm allows the same single gate at many qubit locations using multiple targets
    cqasm_program = """
version 3
qubit[3] q
Z q[0, 1, 2]
"""
    pc = CQASMConverter().convert(cqasm_program, use_postselection=False)
    assert pc.circuit_size == 6

    for _, c in pc.components:
        assert c.name == 'Z'
