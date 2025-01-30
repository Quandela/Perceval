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

from perceval.components import Circuit, Processor, PERM
from perceval.components.core_catalog.postprocessed_cz import PostProcessedCzItem
from .rotations_qloq import G_RHk


def generalized_cz(n: int, m: int) -> Processor:
    """
    Generates a generalized CZ gate for `n` control and `m` target qubits.

    Args:
        n: Number of control qubits.
        m: Number of target qubits.

    Returns:
        Processor: Generalized CZ gate as a Perceval processor object.
    """
    n_modes = 2 ** n
    m_modes = 2 ** m

    total_modes = n_modes + m_modes

    # Identify controls, targets, and swap positions
    control1, control2 = n_modes - 2, n_modes - 1
    target1, target2 = control1 + m_modes, control2 + m_modes

    mapping = {control1: 0, control2: 1, target1: 2, target2: 3}
    circ = Processor("SLOS", total_modes, name="GeneralizedCZ")
    cz = PostProcessedCzItem().build_processor()
    cz.clear_postselection()
    cz.remove_port(0)
    cz.remove_port(2)
    circ.add(mapping, cz)

    return circ

def generate_permutation_for_controlled_op(control: int, target: int, num_qubits: int) -> list[int]:
    """
    Generate the permutation required for a controlled NOT operation

    :param control: Control qubit
    :param target: Target qubit
    :param num_qubits: Number of qubits in this qudit operation.

    :return: the permutation list.
    """
    m = 2 ** num_qubits
    perm = list(range(m))

    for i in range(m):
        binary = format(i, f'0{num_qubits}b')

        if binary[control] == '1':
            flipped = list(binary)
            flipped[target] = '0' if flipped[target] == '1' else '1'
            perm[i] = int("".join(flipped), 2)

    return perm

def create_internal_controlled_op(op_type: str, control: int, target: int, num_qubits: int) -> Circuit:
    """
    Generate a CNOT or CZ gate using the given `control` qubit and `target` qubit for a qudit of size `num_qubits`.

    :param op_type: Operation type (either CNOT or CZ)
    :param control: Control qubit
    :param target: Target qubit
    :param num_qubits: Number of qubits in this qudit operation.

    :return: CNOT or CZ gate, internal to the qudit.
    """
    circ = Circuit(2**num_qubits, name=f"{op_type}")

    # toggling G_RHK makes the below CX into a CZ.
    if op_type == "CZ":
        circ.add(0, G_RHk(num_qubits, target))

    perm = generate_permutation_for_controlled_op(control, target, num_qubits)
    circ.add(0, PERM(perm))  # This is a CX in qudit encoding

    if op_type == "CZ":
        circ.add(0, G_RHk(num_qubits, target))

    return circ

def generate_chained_controlled_ops(op_type: str, n: int) -> Circuit:
    """
    Generates a circuit with a chain of controlled operations.

    Args:
        op_type: Type of controlled operation ("CZ" or "CX").
        n: Number of qubits in the circuit.

    Returns:
        Circuit: A circuit containing the chained controlled operations.
    """
    circ = Circuit(2**n, name=f"{op_type}{n}")

    for i in range(n-1):
        circ.add(0, create_internal_controlled_op(op_type, i, i + 1, n))

    return circ
