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

import numpy as np
from enum import Enum
from .linear_circuit import Circuit
from .unitary_components import BS, PS, PERM


class PauliType(Enum):
    I = 0
    X = 1
    Y = 2
    Z = 3


def get_pauli_circuit(pauli_type: PauliType):
    """
    Create a LO circuit corresponding to one of the Pauli operators (I,X,Y,Z)

    :param pauli_type: PauliType
    :return: 2 mode perceval circuit
    """
    assert isinstance(pauli_type, PauliType), f"Wrong type, expected Pauli, got {type(pauli_type)}"

    if pauli_type == PauliType.I:
        return Circuit(2)
    elif pauli_type == PauliType.X:
        return Circuit(2) // (0, PERM([1, 0]))
    elif pauli_type == PauliType.Y:
        return Circuit(2) // (0, BS.H())
    else:
        return Circuit(2) // (0, BS.H()) // (1, PS(np.pi / 2))


def get_pauli_gate(pauli_type: PauliType):
    """
    Computes one of the Pauli operators (I,X,Y,Z)

    :param pauli_type: PauliType
    :return: 2x2 unitary and hermitian array
    """
    if pauli_type == PauliType.I:
        return np.eye(2, dtype='complex_')
    elif pauli_type == PauliType.X:
        return np.array([[0, 1], [1, 0]], dtype='complex_')
    elif pauli_type == PauliType.Y:
        return np.array([[0, -1j], [1j, 0]], dtype='complex_')
    else:
        return np.array([[1, 0], [0, -1]], dtype='complex_')
