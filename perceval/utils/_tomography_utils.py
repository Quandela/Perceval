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


def state_to_dens_matrix(state):
    return np.dot(state, np.conjugate(np.transpose(state)))


def compute_matrix(j):
    if j == 0:
        return np.eye(2, dtype='complex_')
    elif j == 1:
        return np.array([[0, 1], [1, 0]], dtype='complex_')
    elif j == 2:
        return (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype='complex_')
    elif j == 3:
        return (1 / np.sqrt(2)) * np.array([[1, 1], [1j, -1j]], dtype='complex_')


def matrix_basis(nqubit):  # create a matrix basis from all the tensor products states
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
        M = compute_matrix(k[0])
        for i in k[1:]:
            M = np.kron(compute_matrix(i), M)
        B.append(state_to_dens_matrix(np.dot(M, v)))
    return B


def matrix_to_vector(matrix):  # concatenate a matrix d*d into a vector d**2
    # simply flatten() -> rows after row
    n = len(matrix[0])
    x = np.zeros((n ** 2), dtype='complex_')
    for i in range(n ** 2):
        x[i] = matrix[i // n, i % n]
    return x


def vector_to_matrix(vector):  # expand a vector d**2 into a matrix d*d
    n = len(vector)
    d = int(np.sqrt(n))
    M = np.zeros((d, d), dtype='complex_')
    for i in range(d):
        for j in range(d):
            if len(vector.shape) == 2:
                M[i, j] = vector[i * d + j][0]
            elif len(vector.shape) == 1:
                M[i, j] = vector[i * d + j]
    return M


def decomp(matrix, basis):  # linear decomposition of any matrix upon a basis
    # decomposition used in rho_j creation - process tomography
    n = len(matrix[0])
    y = matrix_to_vector(matrix)
    L = []
    for m in basis:
        L.append(matrix_to_vector(m))
    A = np.zeros((n ** 2, n ** 2), dtype='complex_')
    for i in range(n ** 2):
        for j in range(n ** 2):
            A[i, j] = L[j][i]
    x = np.dot(np.linalg.inv(A), y)
    return x
