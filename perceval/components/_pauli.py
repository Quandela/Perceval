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
from perceval.utils import Matrix


class PauliType(Enum):
    """
    Enumeration of different Pauli  (gates/operators) Types + Identity.
    """
    # Order of members important
    I = 0
    X = 1
    Y = 2
    Z = 3


class PauliEigenStateType(Enum):
    """
    Enumeration of different eigenstates of Pauli operators.
    """
    # Order of members important
    Zm = 0  # Pauli eigen-state Z- : |1>
    Zp = 1  # Pauli eigen-state Z+ : |0>

    Xp = 2  # Pauli eigen-state X+ : |+>
    Xm = 3  # Pauli eigen-state X- : |->

    Yp = 4  # Pauli eigen-state Y+ : |i+>
    Ym = 5  # Pauli eigen-state Y- : |i->


def get_pauli_eigen_state_prep_circ(pauli_type: PauliEigenStateType) -> Circuit:
    """
    Generates a 2-mode LO circuit to prepare the logical states in one of the eigen states of Pauli
    Z_p : |0>, Z_m : |1>, X_p : |+>, X_m : |->, Y_p : |i+>, and Y_m : |i->

    :param pauli_type: PauliType
    :return: 2 mode perceval circuit
    """

    if pauli_type == PauliEigenStateType.Zm:
        return Circuit(2, name="Zm State Preparer")

    elif pauli_type == PauliEigenStateType.Zp:
        return Circuit(2, name="Zp State Preparer") // PERM([1, 0])

    elif pauli_type == PauliEigenStateType.Xp:
        return Circuit(2, name="Xp State Preparer") // BS.H()

    elif pauli_type == PauliEigenStateType.Xm:
        return Circuit(2, name="Xm State Preparer") // PERM([1, 0]) // BS.H()

    elif pauli_type == PauliEigenStateType.Yp:
        return Circuit(2, name="Yp State Preparer") // BS.H() // (1, PS(np.pi / 2))

    elif pauli_type == PauliEigenStateType.Ym:
        return Circuit(2, name="Ym State Preparer") // PERM([1, 0]) // BS.H() // (1, PS(np.pi / 2))

    else:
        raise NotImplementedError(f"{pauli_type}")


def get_pauli_gate(pauli_type: PauliType):
    """
    Uses the PauliType to choose and compute the matrix corresponding Pauli (gates) operators
    (I,X,Y,Z).

    :param pauli_type: PauliType
    :return: 2x2 unitary and hermitian array
    """
    if pauli_type == PauliType.I:
        return Matrix.eye(2)

    elif pauli_type == PauliType.X:
        return Matrix([[0, 1], [1, 0]])

    elif pauli_type == PauliType.Y:
        return Matrix([[0, -1j], [1j, 0]])

    elif pauli_type == PauliType.Z:
        return Matrix([[1, 0], [0, -1]])

    else:
        raise NotImplementedError(f"{pauli_type}")


def get_pauli_eigenvectors(pauli_type) -> list:

    if pauli_type == PauliEigenStateType.Zm:
        return np.array([[1], [0]], dtype='complex_')

    elif pauli_type == PauliEigenStateType.Zp:
        return np.array([[0], [1]], dtype='complex_')

    elif pauli_type == PauliEigenStateType.Xp:
        return (1 / np.sqrt(2)) * np.array([[1], [1]], dtype='complex_')

    elif pauli_type == PauliEigenStateType.Xm:
        return (1 / np.sqrt(2)) * np.array([[1], [-1]], dtype='complex_')

    elif pauli_type == PauliEigenStateType.Yp:
        return (1 / np.sqrt(2)) * np.array([[1], [1j]], dtype='complex_')

    elif pauli_type == PauliEigenStateType.Ym:
        return (1 / np.sqrt(2)) * np.array([[1], [-1j]], dtype='complex_')

    else:
        raise NotImplementedError(f"{pauli_type}")




def get_pauli_eigenvector_matrix(pauli_eigenv) -> np.ndarray:

    if pauli_eigenv == PauliType.X:
        return (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype='complex_')
    elif pauli_eigenv == PauliType.Y:
        return (1 / np.sqrt(2)) * np.array([[1, 1], [1j, -1j]], dtype='complex_')
    else:
        return np.eye((2), dtype='complex_')

def get_pauli_basis_measurement_circuit(pauli_type: PauliType) -> Circuit:
    """
    Uses the PauliType to choose and create LO measurement circuits in Pauli Basis (I,X,Y,Z).

    Equivalent to measuring eigenstates of the 1-qubit Pauli gates

    :param pauli_type: PauliType
    :return: a 2-modes circuit
    """
    assert isinstance(pauli_type, PauliType), f"Wrong type, expected Pauli, got {type(pauli_type)}"

    if pauli_type == PauliType.I:
        return Circuit(2, name="I Measurer")
    elif pauli_type == PauliType.X:
        return Circuit(2, name="Pauli X Measurer") // BS.H()
    elif pauli_type == PauliType.Y:
        return Circuit(2, name="Pauli Y Measurer") // BS.Rx(theta=np.pi/2, phi_bl=np.pi, phi_br=-np.pi/2)
    elif pauli_type == PauliType.Z:
        return Circuit(2, name="Pauli Z Measurer")
    else:
        raise NotImplementedError(f"{pauli_type}")
