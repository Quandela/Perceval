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

from perceval import DensityMatrix, StateVector
from perceval.utils.density_matrix import FockBasis
from typing import Union
import numpy as np
from scipy.sparse import sparray, kron
from scipy import csr_array, dok_array
from math import comb


def extract_upper_triangle(csr_matrix: csr_array) -> csr_array:
    """
    extract the upper half matrix S of a hermitian matrix rho such that S + S_dagger = rho

    """

    result_data = []
    result_indices = []
    result_indptr = [0]
    size = csr_matrix.shape[0]

    for row in range(size):
        start_idx = csr_matrix.indptr[row]
        end_idx = csr_matrix.indptr[row + 1]

        for i in range(start_idx, end_idx):
            col = csr_matrix.indices[i]

            # Only include elements in the upper triangle
            if col > row:
                result_data.append(csr_matrix.data[i])
                result_indices.append(col)
            elif col == row:
                result_data.append(csr_matrix.data[i]/2)
                result_indices.append(col)

        result_indptr.append(len(result_data))

    # Create a new CSR matrix using the extracted upper triangular part
    upper_triangle_matrix = csr_array((result_data, result_indices, result_indptr), shape=csr_matrix.shape)

    return upper_triangle_matrix


def statevector_to_array(sv: StateVector, index: dict):
    """
    translate a StateVector object into an array
    :param sv: a StateVector
    :param index: a dictionary with BasicStates as keys and indices as values
    """
    vector = np.zeros(len(index),  dtype=complex)
    for key, value in sv:
        vector[index[key]] += value
    return vector


def array_to_statevector(vector: Union[np.ndarray, sparray], reverse_index: list):
    """
    translate an array in a StateVector
    :param vector: an array
    :param reverse_index: a list of BasicStates, describing a mapping from indices to the corresponding basic states
    """

    sv = StateVector()
    for i, x in enumerate(reverse_index):
        if vector[i] == 0:
            continue
        else:
            sv += complex(vector[i])*x
    return sv


def density_matrix_tensor_product(A, B):
    """
    Make the tensor product of 2 Density Matrices
    :param A, B: two density matrices
    :return: the "kronecker" product of the density matrices, in the correct basis
    """

    if not isinstance(A, DensityMatrix):
        A = DensityMatrix.from_svd(A)

    if not isinstance(B, DensityMatrix):
        B = DensityMatrix.from_svd(B)

    n_max = A.n_max + B.n_max
    n_mode = A.m + B.m
    size = comb(n_max+n_mode, n_max)
    new_index = FockBasis(n_mode, n_max)
    matrix = kron(A.mat, B.mat)
    perm = dok_array((A.size*B.size, size), dtype=complex)  # matrix from tensor space to complete Fock space
    for i, a_state in enumerate(A.reverse_index):
        for j, b_state in enumerate(B.reverse_index):
            index = new_index[a_state*b_state]
            perm[i*B.size+j, index] = 1

    matrix = perm.T @ matrix @ perm

    return DensityMatrix(matrix, new_index)
