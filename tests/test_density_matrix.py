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
from perceval import StateVector, BasicState, DensityMatrix, Source, SVDistribution
from perceval.utils.density_matrix import create_index, statevector_to_array
import numpy as np
import scipy
from scipy.sparse import dok_array
import pytest
from _test_utils import assert_svd_close


def test_statevector_to_array():
    index = create_index(2, 2)
    sv = StateVector(BasicState([1, 1]))
    vector = np.zeros(6, dtype=complex)
    vector[4] = 1
    assert statevector_to_array(sv, index) == pytest.approx(vector)


def test_create_index():
    dic = create_index(10, 5)
    assert max([basic_state.n for basic_state in dic]) == 5
    for basic_state in dic:
        assert basic_state.m == 10


def test_density_matrix():
    sv = BasicState([0]) + BasicState([1])
    dm = DensityMatrix(sv)
    dm2 = DensityMatrix(BasicState([0]))
    dm3 = dm * 2
    dm4 = 2 * dm

    assert (dm + dm2).size == 2
    assert (dm + dm).size == 2

    for i in range(2):
        for j in range(2):
            assert dm.mat[i, j] == pytest.approx(0.5)
            assert dm3.mat[i, j] == pytest.approx(1)
            assert dm4.mat[i, j] == pytest.approx(1)

    tensor_dm_1 = dm * dm
    tensor_dm_2 = DensityMatrix(sv * sv)
    tensor_dm_3 = dm*sv

    assert tensor_dm_1.shape == (6, 6)
    assert tensor_dm_2.mat.trace() == pytest.approx(1)

    for i in range(tensor_dm_2.size):
        for j in range(tensor_dm_2.size):
            assert tensor_dm_1.mat[i, j] == pytest.approx(tensor_dm_2.mat[i, j])
            assert tensor_dm_1.mat[i, j] == pytest.approx(tensor_dm_3.mat[i, j])

    assert dm[BasicState([0]), BasicState([1])] == pytest.approx(.5)

    assert (dm+dm2).size == 2
    assert (dm+dm).size == 2


def test_density_matrix_to_svd():

    source = Source(.9)
    svd1 = source.generate_distribution(BasicState([0, 1, 0, 1]))
    svd2 = source.generate_distribution(BasicState([0, 1]))
    tensor_svd = svd1*svd2

    dm1 = DensityMatrix(svd1)
    dm2 = DensityMatrix(svd2)

    svd1_back = dm1.to_svd()

    tensor_svd_back = (dm1*dm2).to_svd()
    assert len(svd1_back) == len(svd1)
    assert len(tensor_svd_back) == len(tensor_svd)


def test_density_matrix_array_constructor():
    matrix = np.array([[0.8, 0], [0, 0.2]])
    index = create_index(1, 1)
    dm1 = DensityMatrix(matrix, index)
    dm2 = DensityMatrix(SVDistribution({BasicState([0]): .8, BasicState([1]): .2}))
    assert np.allclose(dm1.mat.toarray(), dm2.mat.toarray())


def test_sample():
    dm = DensityMatrix(BasicState([1]))
    for x in dm.sample(10):
        assert x == BasicState([1])
