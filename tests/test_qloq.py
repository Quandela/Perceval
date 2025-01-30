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

from perceval import PERM, Circuit, catalog
from perceval.components.core_catalog._helpers.entanglement_qloq import generate_permutation_for_controlled_op, \
    create_internal_controlled_op
from perceval.components.core_catalog._helpers.rotations_qloq import internal_swap, G_RHn, G_RHk, _g_rk, _g_rn


def test_internal_swap():
    # Simple swap
    n_qubit = 2

    swap = internal_swap(0, 1, n_qubit)
    assert swap.m == 2 ** n_qubit

    target = PERM([0, 2, 1, 3])
    assert swap.compute_unitary() == pytest.approx(target.compute_unitary())

    # Big swap
    n_qubit = 4

    swap = internal_swap(1, 3, n_qubit)
    assert swap.m == 2 ** n_qubit

    # swap the 2nd and 4th qubit if mode i represents the qubit i written in binary (e.g. 3 --> "0011")
    target = PERM([0, 4, 2, 6, 1, 5, 3, 7, 8, 12, 10, 14, 9, 13, 11, 15])
    assert swap.compute_unitary() == pytest.approx(target.compute_unitary())


def test_rotations_qloq():
    n_qubit = 2
    angle = 1.23
    mapping = {"X": catalog["rx"].build_circuit(theta=angle),
               "Y": catalog["ry"].build_circuit(theta=angle),
               "Z": catalog["rz"].build_circuit(theta=angle),
               "H": catalog["h"].build_circuit()}
    for gate, target in mapping.items():
        # Rotations on last qubit
        if gate == "H":
            circ = G_RHn(n_qubit)
        else:
            circ = _g_rn(gate, angle, n_qubit)
        target_circuit = Circuit(2 ** n_qubit)
        for i in range(0, 2 ** n_qubit, 2):
            target_circuit //= (i, target)

        assert circ.compute_unitary() == pytest.approx(target_circuit.compute_unitary())

        # Now, rotation on 0-th qubit
        if gate == "H":
            circ = G_RHk(n_qubit, 0)
        else:
            circ = _g_rk(angle, n_qubit, 0, gate)

        # Since swap is an involution,
        # applying the swap before and after the circuit should give the same as when we apply the gate on the last qubit
        circ = internal_swap(0, n_qubit - 1, n_qubit) // (0, circ) // (0, internal_swap(0, n_qubit - 1, n_qubit))
        assert circ.compute_unitary() == pytest.approx(target_circuit.compute_unitary())


def test_internal_entanglement():
    # Simple case
    n_qubit = 2

    cnot_perm = generate_permutation_for_controlled_op(0, 1, n_qubit)
    assert cnot_perm == [0, 1, 3, 2]

    cnot_perm = generate_permutation_for_controlled_op(1, 0, n_qubit)
    assert cnot_perm == [0, 3, 2, 1]

    # Big cnot
    n_qubit = 4
    control = 1
    target = 3
    cnot_perm = generate_permutation_for_controlled_op(control, target, n_qubit)
    assert cnot_perm == [0, 1, 2, 3, 5, 4, 7, 6, 8, 9, 10, 11, 13, 12, 15, 14]

    # Now as circuits
    cnot_circ = create_internal_controlled_op("CNOT", control, target, n_qubit)
    assert cnot_circ.compute_unitary() == pytest.approx(PERM(cnot_perm).compute_unitary())

    cz_circ = create_internal_controlled_op("CZ", control, target, n_qubit)
    # Applies H gates to create a CNOT
    cz_circ = G_RHk(n_qubit, target) // (0, cz_circ) // (0, G_RHk(n_qubit, target))

    assert cz_circ.compute_unitary() == pytest.approx(cnot_circ.compute_unitary())
