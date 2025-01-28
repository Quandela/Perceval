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

from perceval.utils.matrix import Matrix

def _count_non_skipped_cols(nb_colums: int, skip_colums: list[int]) -> int:
    return sum([ 1 for i in range(nb_colums) if i not in skip_colums ])

def fidelity(u: Matrix, v: Matrix, skip_colums: list[int] = []) -> float:
    r""" Calculate the fidelity of a unitary implementation compared to a reference unitary

    :param u: the unitary to evaluate
    :param v: the reference unitary
    :skip_colums: list of columns (input modes) that should be ignored when computing fidelity
    :return: real [0-1] float fidelity
    """
    n = _count_non_skipped_cols(u.shape[1], skip_colums)
    if n == 0:
        return 1.0
    f = abs(frobenius_inner_product(u, v, skip_colums)) ** 2 / (n * frobenius_inner_product(u, u, skip_colums))
    if isinstance(f, complex):
        return f.real
    else:
        return f

def modulus_fidelity(u: Matrix, v: Matrix, skip_colums: list[int] = []) -> float:
    r""" Calculate the fidelity of a unitary implementation compared to a reference unitary just comparing
    single input-single output probabilities

    :param u: the unitary to evaluate
    :param v: the reference unitary
    :skip_colums: list of columns (input modes) that should be ignored when computing fidelity
    :return: real [0-1] float fidelity
    """
    n = _count_non_skipped_cols(u.shape[1], skip_colums)
    if n == 0:
        return 1.0
    f = frobenius_inner_product(u, v, skip_colums)/n
    if isinstance(f, complex):
        return f.real
    else:
        return f

def frobenius(u: Matrix, v: Matrix, skip_colums: list[int] = []) -> float:
    r""" Frobenius norm

    :param u: the unitary to evaluate
    :param v: the reference unitary
    :skip_colums: list of columns (input modes) that should be ignored when computing product
    :return: real distance between matrices
    """
    difference = u - v
    return np.sqrt(frobenius_inner_product(difference, difference, skip_colums))

def frobenius_inner_product(A: np.ndarray, B: np.ndarray, skip_colums: list[int] = []) -> float:
    # calculates the inner product associated to Frobenius norm
    C = np.dot(np.transpose(np.conjugate(A)), B)
    # the generic element of C is C_ij = sum(conj(a_ki).b_kj, k=0..N)
    # the i-th diagonal element C_ii is the product of i-th column of A and B
    # C_ii = sum(conj(a_ki).b_ki, k=0..N)
    # as we want to skip cols from A and B, we omit their indices when computing the trace
    return sum([ C.diagonal()[i] for i in range(C.shape[1]) if i not in skip_colums ])
