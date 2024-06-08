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

from abc import abstractmethod
import numpy as np
from perceval.algorithm.abstract_algorithm import AAlgorithm
from perceval.components import AProcessor
from .tomography_utils import _vector_to_sq_matrix, _krauss_repr_ops, _get_canonical_basis_ops


class AProcessTomography(AAlgorithm):
    def __init__(self, processor: AProcessor, **kwargs):
        super().__init__(processor=processor, **kwargs)
        self._nqubit = processor.m // 2
        if self._nqubit > 3:
            raise ValueError(
                f"Input gate too large. Tomography supports up to 3-qubit gates ({self._nqubit}-qubit gate passed).")
        self._size_hilbert = 2 ** self._nqubit

    @abstractmethod
    def chi_matrix(self) -> np.ndarray:
        pass

    def _beta_tensor_elem(self, j: int, k: int, m: int, n: int, nqubit: int) -> np.ndarray:
        """
        computes the elements of beta^{mn}_{jk}, a rank 4 tensor, each index of which can
        take values between 0 and d^2-1  [d = _size_hilbert]

        :param j: one of the indices for the beta tensor, value between 0 and d**2-1
        :param k: one of the indices for the beta tensor, value between 0 and d**2-1
        :param m: one of the indices for the beta tensor, value between 0 and d**2-1
        :param n: one of the indices for the beta tensor, value between 0 and d**2-1
        :param nqubit: numbero f qubits
        :return:
        """

        b = _krauss_repr_ops(m, _get_canonical_basis_ops(j, self._size_hilbert), n, nqubit)
        q, r = divmod(k, self._size_hilbert)  # quotient, remainder
        return b[q, r]

    def _beta_as_matrix(self) -> np.ndarray:
        """
        compiles the 2D beta matrix by extracting elements of the rank 4 tensor computed by method _beta_tensor_elem
        :return: Beta Matrix for Chi computation
        """
        num_meas = self._size_hilbert ** 4  # Total number of measurements needed for process tomography
        beta_matrix = np.zeros((num_meas, num_meas), dtype='complex_')
        for a in range(num_meas):
            j, k = divmod(a, self._size_hilbert ** 2)  # returns quotient, remainder
            for b in range(num_meas):
                # j,k,m,n are indices for _beta_tensor_elem
                m, n = divmod(b, self._size_hilbert ** 2)
                beta_matrix[a, b] = self._beta_tensor_elem(j, k, m, n, self._nqubit)
        return beta_matrix

    def _lambda_target(self, operator: np.ndarray) -> np.ndarray:
        """
        Implements a mathematical formula for ideal gate (given operator) to compute process fidelity
        :param operator: Target operator matrix
        :return: lambda vector to compute chi for the target operator
        """
        lambda_matrix = np.zeros((self._size_hilbert ** 2, self._size_hilbert ** 2), dtype='complex_')
        for j in range(self._size_hilbert ** 2):
            rhoj = _get_canonical_basis_ops(j, self._size_hilbert)
            eps_rhoj = np.linalg.multi_dot([operator, rhoj, np.conjugate(np.transpose(operator))])
            for k in range(self._size_hilbert ** 2):
                quotient, remainder = divmod(k, self._size_hilbert)
                lambda_matrix[j, k] = eps_rhoj[quotient, remainder]

        L1 = np.zeros((self._size_hilbert ** 4, 1), dtype='complex_')
        for i in range(self._size_hilbert ** 4):
            quotient, remainder = divmod(i, self._size_hilbert ** 2)
            L1[i] = lambda_matrix[quotient, remainder]
        return L1

    def chi_target(self, operator: np.ndarray) -> np.ndarray:
        """
        Implements a mathematical formula for ideal gate (given operator) to compute process fidelity
        :param operator: Target operator matrix
        :return: Target Chi matrix
        """

        beta_inv = np.linalg.pinv(self._beta_as_matrix())
        lambd = self._lambda_target(operator)
        X = np.dot(beta_inv, lambd)  # X is a matrix here
        chi = _vector_to_sq_matrix(X[:, 0])
        return chi
