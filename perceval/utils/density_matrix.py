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
from scipy.sparse import csr_array, lil_array, dok_array
import exqalibur as xq


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
                 index: Optional[dict] = None,
                 data_struct: str = "sparse"):
        """
        Constructor for the DensityMatrix Class

        :param mixed_state: SVDistribution, StateVector or Basic State representing a mixed state
        :param index: iterator on all Fock states accessible from this mixed states through a unitary evolution
        """

        if isinstance(mixed_state, (StateVector, BasicState)):
            mixed_state = SVDistribution(mixed_state)

        if not isinstance(mixed_state, SVDistribution):
            raise TypeError("svd must be a BasicState, a StateVector or a SVDistribution")

        self._data_struct = data_struct
        self._m = mixed_state.m
        self._n_max = mixed_state.n_max
        self._size = comb(self.m + self._n_max, self.m)

        if index is None:
            self.index = dict()
            self.reverse_index = []
            k = 0
            for key in max_photon_state_iterator(self._m, self._n_max):
                self.index[key] = k
                self.reverse_index.append(key)
                k+=1
        else:
            self.index = index

        self.mat = dok_array((self._size, self._size), dtype=complex)
        k = 0
        for sv, p in mixed_state.items():
            for bst1 in sv.keys():
                for bst2 in sv.keys():
                    i, j = self.index[bst1], self.index[bst2]
                    self.mat[i, j] += p*sv[bst1]*conj(sv[bst2])
            print(k)
            k+=1

    def __getitem__(self, key):
        """key must be a BasicState tuple"""
        key1, key2 = key
        if not isinstance(key1, BasicState) and isinstance(key2, BasicState):
            raise TypeError("Expected BasicState tuple")
        i, j = self.index[key1], self.index[key2]
        return self.mat[i, j]

    def to_svd(self, threshold=1e-6, n=0):
        """
                gives back an SVDistribution from the density_matrix
        """
        if n == 0:
            n = self.size - 2
        val, vec = scipy.sparse.linalg.eigsh(self.mat, n)
        dic = {}
        for i in range(n):
            if val[i] >= threshold:
                sv = StateVector()
                for j in range(n):
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

        if not self._size == other._size:
            raise ValueError("You can't add Density Matrices with different dimensions")

        if not self._m == other._m:
            raise ValueError("You can't add Density Matrices acting on different numbers of mode")

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
