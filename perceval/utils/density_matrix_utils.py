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

import sys
from typing import Union

import numpy as np
from scipy.sparse import csr_array
if sys.version.startswith('3.8.'):
    from scipy.sparse import spmatrix as sparray
else:
    from scipy.sparse import sparray

from perceval.utils.statevector import StateVector


def extract_upper_triangle(csr_matrix: csr_array) -> csr_array:
    """
    extract sort of the upper triangle of the matrix
    if the input matrix M is hermitian, we have S + S_dagger = M with M the output matrix
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


def is_hermitian(matrix: Union[sparray, np.ndarray]) -> bool:

    n, m = matrix.shape

    if n == m:
        for i in range(n):
            for j in range(i, n):
                if not np.isclose(matrix[i, j], matrix[j, i]):
                    return False
        return True
    return False
