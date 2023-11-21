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
import math
from itertools import product, combinations
from perceval.components import PauliType, get_pauli_gate, get_preparation_circuit


def _state_to_dens_matrix(state: np.ndarray) -> np.ndarray:
    r"""
    computes the density matrix representation of a state
    :math:'$\rho = |\psi> <\psi|$' given a state :math:'$|\psi>$'
    :param state: state of the system
    :return: density matrix
    """
    return np.dot(state, np.conjugate(np.transpose(state)))


def _matrix_basis(nqubit: int, d: int) -> list:
    """
    Generate a list of basis matrices by choosing all combinations of the tensor products of basis states
    :param nqubit: number of qubits
    :param d: Size of hilbert space
    :return: list of basis matrices
    """
    B = []
    pauli_indices = _generate_pauli_index(nqubit)
    v = np.zeros((d, 1), dtype='complex_')
    v[0] = 1

    for elem in pauli_indices:
        M = get_preparation_circuit(elem[0]).compute_unitary()
        if len(elem) > 1:
            M = np.kron(M, get_preparation_circuit(elem[1]).compute_unitary())
        B.append(_state_to_dens_matrix(np.dot(M, v)))

    return B


def _matrix_to_vector(matrix: np.ndarray) -> np.ndarray:
    """
    Reforms a given matrix into a vector by concatenating row wise from the matrix
    :param matrix: any matrix of shape d*d
    :return: a vector of length d**2
    """
    return np.ndarray.flatten(matrix)


def _vector_to_sq_matrix(vector: np.ndarray) -> np.ndarray:
    """
    Reforms a vector of size d**2 into a matrix of size d*d
    :param vector: vector of d**2 length
    :return: Matrix of shape d$d
    """

    size = math.sqrt(len(vector))
    if not size.is_integer():
        raise ValueError("Vector length incompatible to turn into a square matrix")  # checks integer repr of float
    # todo: ugly to do sqrt, remove it and implement param 'd' somewhere consistently
    return np.reshape(vector, (int(size), int(size)))


def _coef_linear_decomp(matrix: np.ndarray, basis: list) -> np.ndarray:
    """
    Solves the linear system of equations : Matrix (M) = Mu x Basis (B) to find Mu.
    "Matrix" can be linearly decomposed in any given "Basis"
    :param matrix: a matrix 'M' to decompose
    :param basis: set of basis matrices 'B' in which the matrix 'M' is to be decomposed
    :return: an array of coefficients 'Mu' which satisfies the equation M = Mu X B
    """

    basis_vectors = np.column_stack([_matrix_to_vector(m) for m in basis])  # convert to list of basis vectors
    y = _matrix_to_vector(matrix)

    # Solve the equation Matrix = Mu x Basis
    mu = np.linalg.solve(basis_vectors, y)
    return mu


def _get_fixed_basis_ops(j: int, nqubit: int) -> np.ndarray:
    """
    computes the set of operators (tensor products of pauli gates) in the fixed basis
    as an array of shape (size_hilbert x size_hilbert)
    :param j: j - number of measurements for state tomography = int between [0,size_hilbert**2 - 1]
    :param nqubit: number of qubits
    :return: operators in a fixed basis (Pauli)
    """
    if nqubit == 1:
        return get_pauli_gate(PauliType(j))

    q, r = divmod(j, (4 ** (nqubit - 1)))
    fix_basis_op = get_pauli_gate(PauliType(q))

    # I am not sure if i completely follow the logic behind what follows. I believe the idea is
    # to keep val between 0-3 no matter the nqubits and value of j. todo: find a better implementation
    # it seems similar to many other calls of creating combinations of operators in a basis
    # but when I called similarly, values for computed fidelity was different

    for i in reversed(range(nqubit - 1)):
        qq, rr = divmod(r, 4**i)
        fix_basis_op = np.kron(fix_basis_op, get_pauli_gate(PauliType(qq)))
        r = r % (4 ** i)  # updating r for next loop - does not do much for nqubit=2 but would affect values >2

    return fix_basis_op


def _get_canonical_basis_ops(j: int, d: int) -> np.ndarray:
    """
    computes the set of operators defined in canonical (standard) basis. Array of size (d x d)
    :param j: number of measurements for state tomography = int between [0, d**2 - 1]
    :param d: size of Hilbert space = 2**nqubits
    :return: operators in a canonical basis
    """
    quotient, remainder = divmod(j, d)
    canonical_op = [[1 if i == quotient and j == remainder else 0 for j in range(d)] for i in range(d)]
    return np.array(canonical_op, dtype='complex_')


def _krauss_repr_ops(m: int, rhoj: np.ndarray, n: int, nqubit: int) -> np.ndarray:
    """
    computes the Krauss representation of an operator = (fixed_basis_op x (canonical_basis_op x fixed_basis_op))
    x: dot product in the previous line
    :param m: indices running between 0 -> d**2 - 1
    :param rhoj: operators in a canonical basis
    :param n: indices running between 0 -> d**2 - 1
    :param nqubit: number of qubits
    :return: Operators in Krauss representation
    """
    return np.dot(_get_fixed_basis_ops(m, nqubit), np.dot(rhoj,
                                                          np.conjugate(np.transpose(_get_fixed_basis_ops(n, nqubit)))))


def _generate_pauli_index(nqubit: int) -> list:
    """
    generates all possible combinations of Pauli indices repeated n times
    :param nqubit: number of qubits
    :return: List of Pauli indices
    """
    s = [pt for pt in PauliType]
    return [list(p) for p in product(s, repeat=nqubit)]  # takes Cartesian product of s with itself repeating 'n' times


def _list_subset_k_from_n(k: int, n: int) -> list:
    """
    list of distinct combination sets of length k from set 's' where 's' is the set {0,...,n-1}
    used only in the method _stokes_parameter
    :param k: index to iterate over the number of qubits
    :param n: number of qubits
    :return: combinations of sets of size k
    """

    s = tuple(range(n))
    return list(combinations(s, k))


def is_physical(input_matrix: np.ndarray, nqubit: int, eigen_tolerance: float = 1e-6) -> dict:
    """
    Verifies if a matrix is trace preserving, hermitian, and completely positive (using the Choi matrix)

    :param input_matrix: chi of a quantum map computed from Quantum Process Tomography
    :param nqubit: Number of qubits
    :param eigen_tolerance: brings a tolerance for the positivity of the eigenvalues of the Choi matrix
    :return: Results about the tests : if Trace Preserving (TP), Hermitian, Completely Positive (CP)
    """
    d2 = len(input_matrix)
    res = {}

    # check if trace preserving
    if not np.isclose(np.trace(input_matrix), 1):
        res['Trace=1'] = False
    else:
        res['Trace=1'] = True

    # check if hermitian
    hermiticity = np.zeros_like(input_matrix).real
    for i in range(d2):
        for j in range(i, d2):
            if not np.isclose(input_matrix[i][j], np.conjugate(input_matrix[j][i])):
                hermiticity[i][j] = 1
    if np.any(hermiticity == 1):
        res['Hermitian'] = False
    else:
        res['Hermitian'] = True

    # check if completely positive with Choi–Jamiołkowski isomorphism
    choi = 0
    for n in range(d2):
        # there were 2 transpose to compute P_n, removed. I do not know if one is important and there is another error
        # Todo : find out about the comment in previous line
        P_n = np.conjugate([_matrix_to_vector(np.transpose(_get_fixed_basis_ops(n, nqubit)))])
        for m in range(d2):
            choi += input_matrix[m, n] \
                    * np.dot(np.transpose([_matrix_to_vector(np.transpose(_get_fixed_basis_ops(m, nqubit)))]), P_n)
    choi /= 2 ** nqubit
    eigenvalues = np.linalg.eigvalsh(choi)
    if np.any(eigenvalues < -eigen_tolerance):
        res['Completely Positive'] = False
    else:
        res['Completely Positive'] = True

    return res
