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
from __future__ import annotations

from perceval.components.core_catalog.gates_1qubit import HadamardItem, RxItem, RyItem, RzItem
from perceval.utils import Parameter
from perceval.components import Circuit, PERM


def internal_swap(qubit1: int, qubit2: int, num_qubits: int) -> Circuit:
    """
    Creates an internal SWAP gate that swaps two qubits within a quantum circuit in a single qudit encoding.

    Args:
        qubit1: The index of the first qubit to be swapped.
        qubit2: The index of the second qubit to be swapped.
        num_qubits: The total number of qubits in the circuit.

    Returns:
        A Perceval circuit object representing the SWAP operation.

    Raises:
        ValueError: If either of the qubit indices are invalid or if they are the same.

    >>> internal_swap(1, 3, 4)
    will swap the second and fourth qubits in a 4-qubit system.

    Note:
        The function constructs a permutation that describes the SWAP operation for the given qubits
    """

    # Check if qubit1 and qubit2 are valid qubit indices
    if qubit1 >= num_qubits or qubit2 >= num_qubits or qubit1 == qubit2:
        raise ValueError("Invalid qubit indices for SWAP.")

    # Calculate the number of possible states
    num_states = 2 ** num_qubits

    # Create a list of all possible states
    states = [bin(i)[2:].zfill(num_qubits) for i in range(num_states)]

    # Generate the swapped states list
    swapped_states = []
    for state in states:
        state_list = list(state)
        # Swap the bits of the two qubits
        state_list[qubit1], state_list[qubit2] = state_list[qubit2], state_list[qubit1]
        swapped_states.append(''.join(state_list))  # Converts back to str

    # Map the states to their indices to get the permutation
    permutation = [states.index(swap_state) for swap_state in swapped_states]

    # Construct the circuit
    circ = Circuit(num_states, name=f"SWAP{qubit1}{qubit2}")
    circ.add(tuple(range(num_states)), PERM(permutation))
    return circ


def _generate_rotation_kth_qubit(gate_layer: Circuit, nqubits: int, k: int, circuit_name: str) -> Circuit:
    """Apply the given gate to the k-th qubit of n qubits for a circuit in a single qudit encoding."""
    if k == nqubits - 1:
        return gate_layer

    circ = Circuit(2 ** nqubits, name=circuit_name)

    # Add the internal swap gate to swap the k-th qubit to the (n-1)-th qubit.
    circ.add(0, internal_swap(k, nqubits - 1, nqubits), merge=True)

    # Apply the rotation gate on the (nqubits-1)-th qubit.
    circ.add(0, gate_layer, merge=True)

    # Revert the modes to their original positions.
    circ.add(0, internal_swap(k, nqubits - 1, nqubits), merge=True)

    return circ


def _generate_rotation_last_qubit(gate: Circuit, nqubits: int) -> Circuit:
    """Apply the gate to nth qubit for a circuit in a single qudit encoding."""
    circ = Circuit(2 ** nqubits, name=gate.name)
    for i in range(0, 2 ** nqubits, 2):
        circ.add(i, gate)
    return circ


def _g_rn(gate_name: str, angle: float | Parameter, n: int ) -> Circuit:
    gates = {"X": RxItem,
             "Y": RyItem,
             "Z": RzItem}

    return _generate_rotation_last_qubit(gates[gate_name]().build_circuit(theta=angle), nqubits=n)


def G_RHn(n: int) -> Circuit:
    """Apply the Hadamard gate to nth qubit for a circuit in a single qudit encoding."""
    return _generate_rotation_last_qubit(HadamardItem().build_circuit(), nqubits=n)


def G_RHk(n: int, k: int) -> Circuit:
    """Apply the Hadamard gate to the k-th qubit of n qubits for a circuit in a single qudit encoding."""
    return _generate_rotation_kth_qubit(G_RHn(n), n, k, f"RH{k}")


def _g_rk(angle: float, n: int, k: int, rotation: str) -> Circuit:
    return _generate_rotation_kth_qubit(_g_rn(rotation, angle, n), n, k, f"R{rotation}{k}")


def apply_rotations_to_qubits(angle_list: list[float | Parameter], n: int, rotation: str):
    """Apply the rotation gate to each qubit in a group of n qubits based on an angle list."""
    assert len(angle_list) == n, "Angle list should match the number of qubits in the group."
    assert rotation in ["Y", "Z", "X"], "Rotation must be X or Y or Z."

    circ = Circuit(2 ** n, name=f"R{rotation}QUDIT{n}")

    # Apply the rotation gate for each qubit based on the provided angle.
    for idx, angle in enumerate(angle_list):
        circ.add(0, _g_rk(angle, n, idx, rotation), merge=True)

    return circ
