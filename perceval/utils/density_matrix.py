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


from statevector import *
from math import comb


class DensityMatrix:
    """Density operator representing a mixed state"""
    def __init__(self, svd: Union[SVDistribution, StateVector, BasicState]):

        if isinstance(svd, (StateVector, BasicState)):
            svd = SVDistribution(svd)

        if not isinstance(svd, SVDistribution):
            raise TypeError("svd must be a BasicState, a StateVector or a SVDistribution")

        self._m = svd.m
        self._n_max = svd.n_max
        self.size = comb(self.m + self._n_max, self.m)

        self.index = dict()
        k = 0
        for key in max_photon_state_iterator(self._m, self._n_max):
            self.index[key] = k
            k+=1

    def __getitem__(self, key):
        key1,key2 = key
        i,j = index[key1], index[key2]
        return self.mat[i,j]

    @property
    def n_max(self):
        return self._n_max

    @property
    def m(self):
        return self._m
