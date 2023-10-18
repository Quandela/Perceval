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


from perceval.utils.statevector import *
from math import comb
from numpy import conj
from scipy.sparse import csr_array, lil_array, dok_array


class DensityMatrixData(csr_array):

    def __getitem__(self, key):
        row, col = key
        if row >= col:
            return super().__getitem__(key)
        else:
            return conj(super().__getitem__(key))

    def __setitem__(self, key, value):
        row, col = key
        if row >= col:
            super().__setitem__(key, value)
        else:
            return conj(super().__setitem__((col,row), value))


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
            k = 0
            for key in max_photon_state_iterator(self._m, self._n_max):
                self.index[key] = k
                k+=1
        else:
            self.index = index

        self.mat = DensityMatrixData((self.size, self.size), dtype=complex)
        for sv, p in svd.items():
            for bst1 in sv.keys():
                for bst2 in sv.keys():
                    if i >= j:
                        i, j = self.index[bst1], self.index[bst2]
                        self.mat[i, j] += p*sv[bst1]*conj(sv[bst2])

    def __getitem__(self, key):
        """key must be a BasicState tuple"""
        key1,key2 = key
        if not (isinstance(key1, BasicState) and isinstance(key2, BasicState)):
            raise TypeError("Expected BasicState tuple")
        i, j = self.index[key1], self.index[key2]
        return self.mat[i, j]

    def to_svd(self):
        pass

    @property
    def n_max(self):
        return self._n_max

    @property
    def m(self):
        return self._m
