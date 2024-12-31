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

from perceval.components import Circuit, PERM, BS


def internal_swap(qubit1: int, qubit2: int, num_qubits: int):
    """
    Creates an internal SWAP gate that swaps two qubits within a quantum circuit.

    Args:
        qubit1: The index of the first qubit to be swapped.
        qubit2: The index of the second qubit to be swapped.
        num_qubits: The total number of qubits in the circuit.

    Returns:
        Circuit: A Perceval circuit object representing the SWAP operation.

    Raises:
        ValueError: If either of the qubit indices are invalid or if they are the same.

    >>> InternalSwap(1, 3, 4)
    will swap the second and fourth qubits in a 4-qubit system.

    Note:
        The function constructs a permutation that describes the SWAP operation for
        the given qubits and adds that permutation to the Perceval circuit.
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


def _generate_rotation_kth_qubit(gate_layer: Circuit, nqubits: int, k: int, circuit_name: str):
    """Apply the Hadamard gate to the k-th qubit of n qubits."""
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


def _generate_rotation_last_qubit(gate: Circuit, nqubits: int, circuit_name: str):
    """Apply the gate to nth qubit."""
    circ = Circuit(2 ** nqubits, name=circuit_name)
    for i in range(0, 2 ** nqubits, 2):
        circ.add(i, gate)
    return circ


def G_RYn(angle, n):
    """Apply the RY gate to nth qubit."""
    return _generate_rotation_last_qubit(
        BS.Ry(theta=angle, phi_tl=0, phi_bl=0, phi_tr=0, phi_br=0), nqubits=n, circuit_name=f"RY{n}")


def G_RYk(angle, n, k):
    """Apply the RY gate to the k-th qubit of n qubits."""
    return _generate_rotation_kth_qubit(G_RYn(angle, n), n, k, f"RY{k}")


def G_RZn(angle, n):
    """Apply the RZ gate to nth qubit."""
    return _generate_rotation_last_qubit(
        BS.Rx(theta=0, phi_tl=-angle / 2, phi_bl=angle / 2, phi_tr=0, phi_br=0), nqubits=n, circuit_name=f"RZ{n}")


def G_RZk(angle, n, k):
    """Apply the RZ gate to the k-th qubit of n qubits."""
    return _generate_rotation_kth_qubit(G_RZn(angle, n), n, k, f"RZ{k}")


def G_RXn(angle, n):
    """Apply the RX gate to nth qubit."""
    return _generate_rotation_last_qubit(BS.Rx(theta=angle), nqubits=n, circuit_name=f"RX{n}")


def G_RXk(angle, n, k):
    """Apply the RX gate to the k-th qubit of n qubits."""
    return _generate_rotation_kth_qubit(G_RXn(angle, n), n, k, f"RX{k}")


def G_RHn(n):
    """Apply the Hadamard gate to nth qubit."""
    return _generate_rotation_last_qubit(BS.H(), nqubits=n, circuit_name=f"RH{n - 1}")


def G_RHk(n, k):
    """Apply the Hadamard gate to the k-th qubit of n qubits."""
    return _generate_rotation_kth_qubit(G_RHn(n), n, k, f"RH{k}")


def apply_rotations_to_qubits(angle_list, n, rotation: str):
    """Apply the RY gate to each qubit in a group of n qubits based on an angle list."""
    assert len(angle_list) == n, "Angle list should match the number of qubits in the group."
    assert rotation in ["Y", "Z", "X"], "Rotation must be X or Y or Z."

    circ = Circuit(2 ** n, name=f"GroupR{rotation}Multi{n}")

    # Apply the rotation gate for each qubit based on the provided angle.
    for idx, angle in enumerate(angle_list):
        if rotation == "X":
            c = G_RXk(angle, n, idx)
        elif rotation == "Y":
            c = G_RYk(angle, n, idx)
        else:
            c = G_RZk(angle, n, idx)
        circ.add(0, c, merge=True)

    return circ
