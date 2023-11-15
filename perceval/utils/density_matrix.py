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
import scipy.sparse.linalg

from perceval.utils.statevector import StateVector, SVDistribution, BasicState, max_photon_state_iterator
from typing import Union, Optional
from math import comb
from numpy import conj
import numpy as np
from scipy.sparse import dok_array, sparray, csr_array, kron
from scipy.sparse.linalg import LinearOperator, eigsh
from scipy.linalg import eigh
from copy import copy
import random


def create_index(m, n_max):
    index = dict()
    for i, x in enumerate(max_photon_state_iterator(m, n_max)):
        index[x] = i
    return index


def statevector_to_array(sv, index):
    """
    translate a StateVector object into an array
    :param sv: a StateVector
    :param index: a dictionary with BasicStates as keys and indices as values
    """
    vector = np.zeros(len(index),  dtype=complex)
    for key, value in sv:
        vector[index[key]] += value
    return vector


def array_to_statevector(vector, reverse_index):
    """
    translate an array in a StateVector
    :param vector: an array
    :param reverse_index: a list of BasicStates
    """

    sv = 0*reverse_index[0]
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

    if isinstance(A, (SVDistribution, StateVector, BasicState)):
        A = DensityMatrix(A)

    if isinstance(B, (StateVector, SVDistribution, BasicState)):
        B = DensityMatrix(B)

    if not isinstance(B, DensityMatrix):
        raise TypeError(f"Cannot do a Tensor product between a DensityMatrix and a {type(B)}")

    if not isinstance(A, DensityMatrix):
        raise TypeError(f"Cannot do a tensor product between a {type(A)} and a DensityMatrix")

    n_max = A.n_max + B.n_max
    n_mode = A.m + B.m
    size = comb(n_max+n_mode, n_max)
    new_index = create_index(n_mode, n_max)
    matrix = kron(A.mat, B.mat)
    perm = dok_array((A.size*B.size, size), dtype=complex) # matrix from tensor space to complete Fock space
    for i, a_state in enumerate(A.reverse_index):
        for j, b_state in enumerate(B.reverse_index):
            index = new_index[a_state*b_state]
            perm[i*B.size+j, index] = 1

    matrix = perm.T @ matrix @ perm

    return DensityMatrix(matrix, new_index)


class DensityMatrix:
    """
    Density operator representing a mixed state
    Does not support annotations
    """
    def __init__(self,
                 mixed_state: Union[SVDistribution, StateVector, BasicState, sparray],
                 index: Optional[dict] = None):
        """
        Constructor for the DensityMatrix Class

        :param mixed_state: 2d-array, SVDistribution, StateVector or Basic State representing a mixed state
        :param index: index of all BasicStates accessible from this mixed states through a unitary evolution
        """
        # Here the constructor for a matrix
        if isinstance(mixed_state, (np.ndarray, sparray)):
            if index is None:
                raise ValueError("you can't construct a DensityMatrix from a matrix without giving an index")
            if len(index) != mixed_state.shape[0]:
                raise ValueError("The index length is incompatible with your matrix size")
            if mixed_state.shape[0] != mixed_state.shape[1]:
                raise ValueError("The density matrix must be square")

            self.mat = csr_array(mixed_state, dtype=complex)
            self._size = self.mat.shape[0]
            self._m = next(iter(index.keys())).m
            self._n_max = max([x.n for x in index.keys()])
            self.index = dict()
            self.reverse_index = []
            self.set_index(index)  # index construction

        else:
            # Here the constructor for an SVD, SV or BS
            if isinstance(mixed_state, (StateVector, BasicState)):
                mixed_state = SVDistribution(mixed_state)

            if not isinstance(mixed_state, SVDistribution):
                raise TypeError("svd must be a BasicState, a StateVector a SVDistribution or a 2d array")

            self._m = mixed_state.m
            self._n_max = mixed_state.n_max
            self._size = comb(self.m + self._n_max, self.m)

            self.index = dict()
            self.reverse_index = []
            self.set_index(index)  # index construction

            # matrix construction from svd
            l = []
            for sv, p in mixed_state.items():
                vector = dok_array((self._size, 1), dtype=complex)
                for bst in sv.keys():
                    idx = self.index[bst]
                    vector[idx, 0] = sv[bst]
                vector = csr_array(vector)
                l.append((vector, p))
            self.mat = sum([p*(vector @ conj(vector.T)) for vector, p in l])

    def set_index(self, index):
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

    def __getitem__(self, key):
        """key must be a BasicState tuple"""
        key1, key2 = key
        if not isinstance(key1, BasicState) and isinstance(key2, BasicState):
            raise TypeError("Expected BasicState tuple")
        i, j = self.index[key1], self.index[key2]
        return self.mat[i, j]

    @staticmethod
    def _deflation(A, val, vec):
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

        def matrix_vector_multiplication(x):
            """
            The matrix vector multiplication function associated to the deflated operator
            """
            result = A @ x
            for k in range(val.shape[0]):
                result -= val[k] * (conj(vec[:, k].T) @ x) * vec[:, k]

            return result

        return matrix_vector_multiplication

    @staticmethod
    def _bra_str(bs):
        return "<" + str(bs)[1:][:-1] + "|"

    def to_svd(self, threshold=1e-8, batch_size=1):
        """
                gives back an SVDistribution from the density_matrix
        """
        if self.size < 50:  # if the matrix is small: array eigh method
            matrix = self.mat.toarray()
            w, v = eigh(matrix)
            dic = {}
            for k in range(w.shape[0]):
                sv = array_to_statevector(v[:, k], self.reverse_index)
                if w[k] > threshold:
                    dic[sv] = w[k]
        else:  # if the matrix is large: sparse eigsh method
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
                        sv += complex(vec[j][i])*self.reverse_index[j]
                    sv.normalize()
                    if sv.m != 0:
                        dic[sv] = val[i]
                else:
                    continue
        return SVDistribution(dic)

    def __add__(self, other):
        """
        add two density Matrices together
        """

        if not isinstance(other, DensityMatrix):
            raise TypeError("You can only add a Density Matrix to a Density Matrix")

        if not self._m == other._m:
            raise ValueError("You can't add Density Matrices acting on different numbers of mode")

        n = max(self._n_max, other.n_max)

        if n == self.n_max:
            copy_mat = copy(other.mat)
            copy_mat.resize(self._size, self._size)
            new_mat = copy_mat + self.mat
            new_index = self.index
        else:
            copy_mat = copy(self.mat)
            copy_mat.resize(other._size, other._size)
            new_mat = copy_mat + other.mat
            new_index = other.index

        return DensityMatrix(new_mat, new_index)

    def __mul__(self, other):
        """
        Make a tensor product between a Density Matrix and a mixed state in any form
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

    def normalize(self):
        """
        Normalize the density matrix so that Trace(\rho) = 1
        """

        factor = self.mat.trace()
        self.mat = (1/factor)*self.mat

    def sample(self, count=1):
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
        str(self.mat.toarray())

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
