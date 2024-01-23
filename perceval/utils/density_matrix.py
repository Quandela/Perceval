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

from perceval.utils.statevector import SVDistribution, BasicState, max_photon_state_iterator
from perceval.utils.density_matrix_utils import array_to_statevector
from typing import Union, Optional, Tuple
from math import comb
from numpy import conj
import numpy as np
from scipy.sparse import dok_array, sparray, csr_array, kron
from scipy.sparse.linalg import LinearOperator, eigsh
from scipy.linalg import eigh
from copy import copy
import random
import exqalibur as xq

# In all the DensityMatrix Class, there is a compromise between csr_array and dok_array.
# The first one is well suited for matrix-vector product, the other one is easier to construct from scratch


class FockBasis(dict):
    def __init__(self, m, n_max):
        for i, st in enumerate(max_photon_state_iterator(m, n_max)):
            self[st] = i
        self._m = m
        self._n_max = n_max

    def add_photon(self):
        self._n_max += 1
        length = len(self)
        new_states = xq.FSArray(self._m, self._n_max)
        for i, st in enumerate(new_states):
            self[st] = length+i

    def add_photons(self, n):
        for _ in range(n):
            self.add_photon()

    @property
    def m(self):
        return self._m

    @property
    def n_max(self):
        return self._n_max


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


class DensityMatrix:
    """
    Density operator representing a mixed state
    Does not support annotations yet
    """
    def __init__(self,
                 mixed_state: Union[np.array, sparray],
                 index: Optional[FockBasis] = None,
                 m: Optional[int] = None,
                 n_max: Optional[int] = None):
        """
        Constructor for the DensityMatrix Class

        :param mixed_state: 2d-array, SVDistribution, StateVector or Basic State representing a mixed state
        :param index: index of all BasicStates accessible from this mixed states through a unitary evolution
        :param m: optional number of modes if index is not given
        :param n_max: optional maximum number of photon if index is not given
        """
        # Here the constructor for a matrix
        if not isinstance(mixed_state, (np.ndarray, sparray)):
            raise TypeError(f"Can't construct a density matrix from {type(mixed_state)}")
        if index is None:
            if not (m is None or n_max is None):
                index = FockBasis(m, n_max)
            else:
                raise ValueError("you must provide an index or a number of modes and photons")
        if not isinstance(index, FockBasis):
            raise ValueError(f"index must be a FockBasis object. {type(index)} was given")
        if len(index) != mixed_state.shape[0]:
            raise ValueError(f"The index length is incompatible with your matrix size. \n "
                             f"For at most {index.n_max} photons in {index.m} modes, your matrix size must be {len(index)}")
        if mixed_state.shape[0] != mixed_state.shape[1]:
            raise ValueError("The density matrix must be square")

        self.mat = csr_array(mixed_state, dtype=complex)
        self._size = self.mat.shape[0]
        self._m = index.m
        self._n_max = index.n_max

        self.index = dict()
        self.reverse_index = []
        self.set_index(index)  # index construction

    @staticmethod
    def from_svd(svd: Union[SVDistribution, StateVector, BasicState], index: Optional[FockBasis] = None):
        """
        Construct a Density matrix from a SVDistribution
        :param svd: an SVDistribution object representing the mixed state
        :param index: the basis in which the density matrix is expressed. Self generated if incorrect
        :return: the DensityMatrix object corresponding to the SVDistribution given
        """
        if isinstance(svd, (StateVector, BasicState)):
            svd = SVDistribution(svd)

        if not isinstance(svd, SVDistribution):
            raise TypeError("mixed_state must be a BasicState, a StateVector a SVDistribution or a 2d array")

        for key in svd.keys():
            if any([bs[0].has_annotations for bs in key]):
                raise ValueError("annotations are not supported yet in DensityMatrix")

        m = svd.m
        n_max = svd.n_max
        size = comb(m+n_max, m)

        if not(isinstance(index, FockBasis) and index.m == m and index.n_max >= n_max):
            index = FockBasis(m, n_max)
        l = []
        for sv, p in svd.items():
            vector = np.zeros((size, 1), dtype=complex)
            for bst in sv.keys():
                idx = index[bst]
                vector[idx, 0] = sv[bst]
            vector = csr_array(vector)
            l.append((vector, p))
        matrix = sum([p * (vector @ conj(vector.T)) for vector, p in l])

        return DensityMatrix(matrix, index)

    def set_index(self, index: dict):
        if index is None:
            k = 0
            for key in max_photon_state_iterator(self._m, self._n_max):
                self.index[key] = k
                self.reverse_index.append(key)
                k += 1
        else:
            if len(index) == self._size:
                self.index = index
                self.reverse_index = [None]*self._size
                for key in index.keys():
                    self.reverse_index[index[key]] = key
            else:
                raise ValueError("the index size does not match the matrix size")

    def __getitem__(self, key: Tuple[BasicState, BasicState]):
        """key must be a BasicState tuple"""
        key1, key2 = key
        if not isinstance(key1, BasicState) or not isinstance(key2, BasicState):
            raise TypeError("Expected BasicState tuple")
        i, j = self.index[key1], self.index[key2]
        return self.mat[i, j]

    @staticmethod
    def _deflation(A: sparray, val: np.ndarray, vec: np.ndarray):
        """
        Defines the mat_vec function of the Linear operator after the deflation of all the vectors in the vec array
        :param A: any kind of sparse matrix
        :param val: the array of eigen_values
        :param vec: the array of eigen_vector
        """
        if val.shape[0] != vec.shape[1]:
            raise ValueError("inconsistent number of eigenvectors and eigenvalues")

        if vec.shape[0] != A.shape[0]:
            raise ValueError("the size of the matrix is inconsistent with this of the eigenvector")

        def matrix_vector_multiplication(x: np.ndarray):
            """
            The matrix vector multiplication function associated to the deflated operator
            """
            result = A @ x
            for k in range(val.shape[0]):
                result -= val[k] * (conj(vec[:, k].T) @ x) * vec[:, k]

            return result

        return matrix_vector_multiplication

    @staticmethod
    def _bra_str(bs: BasicState):
        return "<" + str(bs)[1:][:-1] + "|"

    def _to_svd_small(self, threshold):
        """Extracting the svd using the scipy method on arrays"""
        matrix = self.mat.toarray()
        w, v = eigh(matrix)
        dic = {}
        for k in range(w.shape[0]):
            sv = array_to_statevector(v[:, k], self.reverse_index)
            if w[k] > threshold:
                dic[sv] = w[k]
        return SVDistribution(dic)

    def _to_svd_large(self, threshold, batch_size):
        """Extracting the svd using the scipy method on sparse matrices"""
        val = np.array([])
        vec = np.empty(shape=(self._size, 0), dtype=complex)
        while (val > threshold).all():
            deflated_operator = LinearOperator((self._size, self._size), matvec=self._deflation(self.mat, val, vec))
            new_val, new_vec = eigsh(deflated_operator, batch_size)
            val = np.concatenate((val, new_val))
            vec = np.concatenate((vec, new_vec), axis=1)
        dic = {}
        for i in range(val.shape[0]):
            if val[i] >= threshold:
                sv = StateVector()
                for j in range(len(vec)):
                    sv += complex(vec[j][i]) * self.reverse_index[j]
                sv.normalize()
                if sv.m != 0:
                    dic[sv] = val[i]
            else:
                continue
        return SVDistribution(dic)

    def to_svd(self, threshold: float = 1e-8, batch_size: int = 1):
        """
            gives back an SVDistribution from the density_matrix
            :param threshold: the threshold when the search for eigen values is stopped
            :param batch_size: the number of eigen values at each Arnoldi's algorithm iteration.
                Only used if matrix is large enough.
            :return: The SVD object corresponding to the DensityMatrix.
                The StateVector with probability < threshold are removed.
        """
        if self.size < 50:  # if the matrix is small: array eigh method
            return self._to_svd_small(threshold)

        else:  # if the matrix is large: sparse eigsh method
            return self._to_svd_large(threshold, batch_size)

    def __add__(self, other):
        """
        add two density Matrices together
        """

        if not isinstance(other, DensityMatrix):
            raise TypeError("You can only add a Density Matrix to a Density Matrix")

        if self._m != other._m:
            raise ValueError("You can't add Density Matrices with different numbers of mode")

        n = max(self._n_max, other.n_max)
        if n == self.n_max:
            small_matrix = other
            big_matrix = self
        else:
            small_matrix = self
            big_matrix = other
        copy_mat = copy(small_matrix.mat)
        copy_mat.resize(big_matrix.size, big_matrix.size)
        new_mat = copy_mat + big_matrix.mat
        new_index = big_matrix.index

        return DensityMatrix(new_mat, new_index)

    def __mul__(self, other):
        """
            Make a tensor product between a Density Matrix and a mixed state in any form
            Or make a scalar multiplication
        """

        if isinstance(other, (int, float, complex)):
            new_dm = copy(self)
            new_dm.mat = other*new_dm.mat
            return new_dm

        return density_matrix_tensor_product(self, other)

    def __rmul__(self, other):
        """
            Make a tensor product between a mixed in any form and a Density Matrix
            Or make a scalar multiplication
        """
        if isinstance(other, (int, float, complex)):
            new_dm = copy(self)
            new_dm.mat = other*new_dm.mat
            return new_dm

        else:
            return density_matrix_tensor_product(other, self)

    def remove_low_amplitude(self, threshold=1e-6):
        """
        Remove the lines and column where the amplitude is below a certain threshold
        """
        projector = dok_array(self.shape)
        for k in range(self.size):
            if self.mat[k, k] > threshold:
                projector[k, k] = 1

        projector = csr_array(projector)

        self.mat = projector.dot(self.mat).dot(projector)
        self.normalize()

    def normalize(self):
        """
        Normalize the density matrix so that Trace(\rho) = 1
        """

        factor = self.mat.trace()
        self.mat = (1/factor)*self.mat

    def sample(self, count: int = 1):
        """
        sample on the density matrix
        """
        self.normalize()
        output = random.choices(self.reverse_index, list(self.mat.diagonal()), k=count)
        return output

    def __str__(self):
        """
        string representation of a density matrix
        """
        string = ""
        for i in range(self._size):
            for j in range(self._size):
                if self.mat[i, j] == 0:
                    continue
                else:
                    new_term = (f"{self.mat[i, j]:.2f}*" +
                                str(self.reverse_index[j]) +
                                self._bra_str(self.reverse_index[i]) +
                                "+")
                    string += new_term

        return string[:-1]

    def __repr__(self):
        return str(self.mat.toarray())

    @property
    def n_max(self):
        return self._n_max

    @property
    def m(self):
        return self._m

    @property
    def shape(self):
        return self._size, self._size

    @property
    def size(self):
        return self._size
