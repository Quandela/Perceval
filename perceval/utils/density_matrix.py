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

from perceval.utils.statevector import *
from math import comb
from numpy import conj
import numpy as np
from scipy.sparse import dok_array, sparray
from scipy.sparse.linalg import LinearOperator, eigsh
import exqalibur as xq


def create_index(m, n_max):
    l = dict()
    for i, x in enumerate(max_photon_state_iterator(m, n_max)):
        l[x]=i
    return l


def density_matrix_tensor_product(A, B):
    """
    Make the tensor product of 2 Density Matrices
    :param A, B: two density matrices
    :return: the "kronecker" product of the density matrices
    """
    n_max = A.n_max + B.n_max
    n_mode = A.m + B.n_max
    size = comb(n_max+n_mode, n_max)
    new_index = create_index(n_mode, n_max)
    matrix = dok_array((size, size), dtype="complex")

    new_dm = DensityMatrix(matrix, new_index)
    for ia,ja in A.mat.keys():
        ket_a, bra_a = A.reverse_index(ia), A.reverse_index(ja)
        for ib,jb in B.mat.keys():
            ket_b, bra_b = B.reverse_index[ib], B.reverse_index(jb)
            new_dm[ket_a*ket_b, bra_a*bra_b] = A.mat[ia, ja]*A.mat[ib, jb]

    return new_dm


class DiagonalBlockMatrix:
    """
    A DataType for large block diagonal matrices
    Specially designed for Density matrices with no superposition between void and photons

    """

    def __init__(self, n_max, m):
        """
        initiate a zero square matrix for n_max photons and m modes
        """

        self.data = []
        self.block_indices = [0]
        for k in range(n_max + 1):
            dim = comb(m+k-1, k)
            self.data.append(np.zeros((dim, dim), dtype=complex))
            self.block_indices.append(self.block_indices[-1] + dim)

    def __getitem__(self, item):
        i, j = item
        bloc_idx1, bloc_idx2, k, l = self.get_indices(i,j)
        if bloc_idx2 != bloc_idx1:
            return 0j
        else:
            return self.data[bloc_idx1 - 1][k, l]

    def __setitem__(self, key, value):
        i, j = key
        bloc_idx1, bloc_idx2, k, l = self.get_indices(i, j)
        if bloc_idx2 != bloc_idx1:
            raise IndexError(f"Can't affect non zero values at these indices: {(i,j)}")
        else:
            self.data[bloc_idx1 - 1][k, l] = value

    def __rmul__(self, other):
        for i, x in enumerate(self.data):
            self.data[i] = other*x


    def get_indices(self, i, j):
        bloc_idx1, bloc_idx2 = 0, 0
        while bloc_idx1 < i:
            bloc_idx1 += 1
        while bloc_idx2 < j:
            bloc_idx2 += 1
        return bloc_idx1, bloc_idx2, i-bloc_idx1, j-bloc_idx2

    @property
    def shape(self):
        return


class DensityMatrix:
    """
    Density operator representing a mixed state
    Does not support annotations
    """
    def __init__(self,
                 mixed_state: Union[SVDistribution, StateVector, BasicState],
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

            self.mat = dok_array(mixed_state)
            self._size = self.mat.shape[0]
            self._m = next(iter(index.keys()))
            self._n_max = max([x.n for x in index.keys()])
            self.index = dict()
            self.reverse_index = []
            self.set_index(index)  # index construction
            self.is_block_diagonal = False

        else:
            # Here the constructor for an SVD, SV or BS
            if isinstance(mixed_state, (StateVector, BasicState)):
                mixed_state = SVDistribution(mixed_state)

            if not isinstance(mixed_state, SVDistribution):
                raise TypeError("svd must be a BasicState, a StateVector a SVDistribution or a 2d array")

            self._m = mixed_state.m
            self._n_max = mixed_state.n_max
            self._size = comb(self.m + self._n_max, self.m)
            self.is_block_diagonal = True

            self.block_index = [0]
            space_size = 1
            for n_photon in range(self._n_max):
                self.block_index.append(self.block_index[-1] + space_size)
                space_size = (space_size*(self._m+n_photon))//(n_photon+1)
                print(space_size)

            print("done1")

            self.index = dict()
            self.reverse_index = []
            self.set_index(index)  # index construction

            print("done2")

            self.mat = dok_array((self._size, self._size), dtype=complex)

            # matrix construction from svd
            l=[]
            for sv, p in mixed_state.items():
                if len(sv.n) > 1:
                    self.is_block_diagonal = False
                vect = dok_array((self._size, 1), dtype=complex)
                for bst in sv.keys():
                    idx = self.index[bst]
                    vect[idx,0] = sv[bst]
                l.append((vect, p))

            for x in l:
                vect, p = x
                self.mat += p*(vect @ conj(vect.T))

    def set_index(self, index):
        if index is None:
            k = 0
            for key in max_photon_state_iterator(self._m, self._n_max):
                self.index[key] = k
                self.reverse_index.append(key)
                k+=1
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
        Defines the mat_vec function of the Linear operator after the deflation of all the vectors in eigen_vec_list
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

    def to_svd(self, threshold=1e-8, batch_size=1):
        """
                gives back an SVDistribution from the density_matrix
        """
        val = np.array([])
        vec = np.empty(shape=(self._size, 0), dtype=complex)
        k=0
        while (val > threshold).all():
            if val.shape[0] > 0:
                print(val[-1])
            deflated_operator = LinearOperator((self._size, self._size), matvec=self._deflation(self.mat, val, vec))
            new_val, new_vec = eigsh(deflated_operator, batch_size)
            val = np.concatenate((val, new_val))
            vec = np.concatenate((vec, new_vec), axis=1)

        dic = {}
        print(len(val))
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
        m = self._m
        if n == self.n_max:
            new_mat = copy(self.mat)
            other_mat = other.mat
            new_index = self.index
        else:
            new_mat = copy(other.mat)
            other_mat = self.mat
            new_index = other.index

        for key, value in other_mat.items():
            new_mat[key] += value

        return DensityMatrix(new_mat, new_index)

    def __mul__(self, other):
        """
        Make a tensor product between a Density Matrix and a mixed state in any form
        """

        if isinstance(other, (int, float, complex)):
            new_dm = copy(self)
            new_dm.mat = other*new_dm.mat
            return new_dm

        if isinstance(other, (SVDistribution, StateVector, BasicState)):
            other = DensityMatrix(other)

        if not isinstance(other, DensityMatrix):
            raise TypeError(f"Cannot do a Tensor product between a DensityMatrix and a {type(other)}")

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

        if isinstance(other, (StateVector, SVDistribution, BasicState)):
            other = DensityMatrix(other)

        if not isinstance(other, DensityMatrix):
            raise TypeError(f"Cannot do a tensor product between a {type(other)} and a DensityMatrix")
        else:
            return density_matrix_tensor_product(other, self)

    def normalize(self):
        """
        Normalize the density matrix so that Trace(\rho) = 1
        """

        factor = self.mat.trace()
        self.mat = (1/factor)*self.mat

    @property
    def n_max(self):
        return self._n_max

    @property
    def m(self):
        return self._m

    @property
    def shape(self):
        return self._size, self._size
