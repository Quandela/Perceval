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

class DensityMatrix:
    """
    Density operator representing a mixed state
    Does not support annotations
    """
    def __init__(self, svd: Union[SVDistribution, StateVector, BasicState], index: Optional[dict] = None):
        """
        Constructor for the DensityMatrix Class

        :param svd: SVDistribution, StateVector or Basic State representing a mixed state
        :param index: iterator on all Fock states accessible from this mixed states through a unitary evolution
        """

        if isinstance(svd, (StateVector, BasicState)):
            svd = SVDistribution(svd)

        if not isinstance(svd, SVDistribution):
            raise TypeError("svd must be a BasicState, a StateVector or a SVDistribution")

        self._m = svd.m
        self._n_max = svd.n_max
        self.size = comb(self.m + self._n_max, self.m)
        if index is None or len(index != self.size):
            self.index = dict()
            self.reverse_index = []
            k = 0
            for key in max_photon_state_iterator(self._m, self._n_max):
                self.index[key] = k
                self.reverse_index.append(key)
                k+=1
        else:
            self.index = index

        print("index constructed")
        k = 0
        self.mat = dok_array((self.size, self.size), dtype=complex)
        for sv, p in svd.items():
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

    def normalize(self):
        """
        Normalize the density matrix so that Tr(\rho) = 1
        """

        factor = self.mat.trace()
        self.mat = (1/factor)*dm.mat


    @property
    def n_max(self):
        return self._n_max

    @property
    def m(self):
        return self._m

    @property
    def shape(self):
        return self.size, self.size
