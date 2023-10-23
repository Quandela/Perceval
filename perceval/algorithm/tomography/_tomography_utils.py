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
from perceval.components import PauliType, get_pauli_gate


def _state_to_dens_matrix(state):
    return np.dot(state, np.conjugate(np.transpose(state)))


def _compute_matrix(j):
    if j == 0:
        return np.eye(2, dtype='complex_')
    elif j == 1:
        return np.array([[0, 1], [1, 0]], dtype='complex_')
    elif j == 2:
        return (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype='complex_')
    elif j == 3:
        return (1 / np.sqrt(2)) * np.array([[1, 1], [1j, -1j]], dtype='complex_')


def _matrix_basis(nqubit):  # create a matrix basis from all the tensor products states
    # needed for rho_j and to compute epsilon rho_j
    d = 2 ** nqubit
    B = []
    for j in range(d ** 2):
        v = np.zeros((d, 1), dtype='complex_')
        v[0] = 1
        k = []
        for m in range(nqubit - 1, -1, -1):
            k.append(j // (4 ** m))
            j = j % (4 ** m)
        k.reverse()
        M = _compute_matrix(k[0])
        for i in k[1:]:
            M = np.kron(_compute_matrix(i), M)
        B.append(_state_to_dens_matrix(np.dot(M, v)))
    return B


def _matrix_to_vector(matrix):
    # Returns a vector (size=>d**2) given a matrix (shape=>d*d) -> concatenates row wise
    return np.ndarray.flatten(matrix)


def _vector_to_sq_matrix(vector):
    # expand a vector d**2 into a matrix d*d
    size = math.sqrt(len(vector))
    if not size.is_integer():
        raise ValueError("Vector length incompatible to turn into a square matrix")  # checks integer repr of float
    # todo: ugly to do sqrt, remove it and implement param 'd' somewhere consistently
    return np.reshape(vector, (int(size), int(size)))


def _decomp(matrix, basis):  # linear decomposition of any matrix upon a basis
    # decomposition used in rho_j creation - process tomography
    n = len(matrix[0])
    y = _matrix_to_vector(matrix)
    L = []
    for m in basis:
        L.append(_matrix_to_vector(m))
    A = np.zeros((n ** 2, n ** 2), dtype='complex_')
    for i in range(n ** 2):
        for j in range(n ** 2):
            A[i, j] = L[j][i]
    x = np.dot(np.linalg.inv(A), y)
    return x


def _get_fixed_basis_ops(j, nqubit):
    """
    computes the fixed sets of operators (tensor products of pauli gates) for quantum process tomography

    :param j: int between 0 and 4**nqubit-1
    :param nqubit: number of qubits
    :return: 2**nqubit x 2**nqubit array
    """
    if nqubit == 1:
        return get_pauli_gate(PauliType(j))

    # pauli_indices = generate_pauli_index(nqubit)
    # print(pauli_indices)
    # for val in pauli_indices:

    index = j // (4 ** (nqubit - 1))  # todo: fix 'index' and 'val' in counting
    fix_basis_op = get_pauli_gate(PauliType(index))

    j = j % (4 ** (nqubit - 1))
    for i in range(nqubit - 2, -1, -1):
        val = j // (4 ** i)
        fix_basis_op = np.kron(fix_basis_op, get_pauli_gate(PauliType(val)))
        j = j % (4 ** i)
    return fix_basis_op


def _get_canonical_basis_ops(j, nqubit):
    """
    Computes the matrices of the canonical basis

    :param j: int between 0 and 4**nqubit-1
    :param nqubit: number of qubits
    :return: 2**nqubit x 2**nqubit array
    """
    d = 2 ** nqubit
    canonical_op = np.zeros((d, d), dtype='complex_')
    canonical_op[j // d, j % d] = 1
    return canonical_op


def _krauss_repr_ops(m, rhoj, n, nqubit):
    # computes the Krauss representation of the operator
    # Returns (fixed_basis_op x (canonical_basis_op x fixed_basis_op)). x: dot product here
    return np.dot(_get_fixed_basis_ops(m, nqubit), np.dot(rhoj, np.conjugate(np.transpose(_get_fixed_basis_ops(n, nqubit)))))


def _generate_pauli_index(n):
    S = [PauliType(member.value) for member in PauliType]  # todo : maybe a better way exists
    # takes Cartesian product of elements of set S with itself repeating 'n' times
    output_set = [list(p) for p in product(S, repeat=n)]
    return output_set


def _list_subset_k_from_n(k, n):
    # list of distinct combination sets of length k from set 's' where 's' is the set {0,...,n-1}
    # used only in the method _stokes_parameter
    #  Should we put it in overall utils? or have a specific util for tomograph?
    s = tuple(range(n))
    return list(combinations(s, k))
