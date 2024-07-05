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

import math
import pytest

import numpy as np

from perceval import BasicState, Source, SVDistribution, Matrix, Simulator, \
                      Unitary, Circuit, SLOSBackend
from perceval.utils.density_matrix import FockBasis, DensityMatrix
from perceval.utils.density_matrix_utils import *



def test_fock_basis():

    basis = FockBasis(12,3)
    assert basis.m == 12
    assert len(basis) == 455
    basis.add_photon()
    assert len(basis) == 1820
    basis.add_photons(2)
    assert len(basis) == 18564
    assert basis.n_max == 6


def test_statevector_to_array():
    index = FockBasis(2, 2)
    sv = StateVector(BasicState([1, 1]))
    vector = np.zeros(6, dtype=complex)
    vector[4] = 1
    assert statevector_to_array(sv, index) == pytest.approx(vector)


def test_create_index():
    dic = FockBasis(10, 5)
    assert max([basic_state.n for basic_state in dic]) == 5
    for basic_state in dic:
        assert basic_state.m == 10


def test_density_matrix():
    sv = BasicState([0]) + BasicState([1])
    dm = DensityMatrix.from_svd(sv)
    dm2 = DensityMatrix.from_svd(BasicState([0]))
    dm3 = dm * 2
    dm4 = 2 * dm

    assert (dm + dm2).size == 2
    assert (dm + dm).size == 2

    test_mat = np.ones((2,2))
    assert np.allclose(dm.mat.toarray(), 0.5*test_mat, 1e-6, 1e-6)
    assert np.allclose(dm3.mat.toarray(), test_mat, 1e-6, 1e-6)
    assert np.allclose(dm4.mat.toarray(), test_mat, 1e-6, 1e-6)

    tensor_dm_1 = dm * dm
    tensor_dm_2 = DensityMatrix.from_svd(sv * sv)
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

    dm1 = DensityMatrix.from_svd(svd1)
    dm2 = DensityMatrix.from_svd(svd2)

    svd1_back = dm1.to_svd()

    tensor_svd_back = (dm1*dm2).to_svd()
    assert len(svd1_back) == len(svd1)
    assert len(tensor_svd_back) == len(tensor_svd)


def test_density_matrix_array_constructor():
    matrix = np.array([[0.8, 0], [0, 0.2]])
    index = FockBasis(1, 1)
    dm1 = DensityMatrix(matrix, index)
    dm2 = DensityMatrix.from_svd(SVDistribution({BasicState([0]): .8, BasicState([1]): .2}))
    dm3 = DensityMatrix(matrix, m=1, n_max=1)
    assert np.allclose(dm1.mat.toarray(), dm3.mat.toarray())
    assert np.allclose(dm1.mat.toarray(), dm2.mat.toarray())


def test_sample():
    dm = DensityMatrix.from_svd(BasicState([1]))
    for x in dm.sample(10):
        assert x == BasicState([1])


def test_avoid_annotations():
    with pytest.raises(ValueError):
        DensityMatrix.from_svd(BasicState('|{_:1}>')+BasicState('|{_:2}>'))


def test_remove_low_amplitude():

    s = Source(.991)
    dm = DensityMatrix.from_svd(s.generate_distribution(BasicState([1, 0, 1, 0, 1, 0])))

    assert dm.mat.nnz == 8
    assert dm.mat.trace() == pytest.approx(1)

    dm.remove_low_amplitude()

    assert dm.mat.nnz == 7
    assert dm.mat.trace() == pytest.approx(1)
    assert dm[BasicState([0, 0, 0, 0, 0, 0]), BasicState([0, 0, 0, 0, 0, 0])] == 0

    dm.remove_low_amplitude(1e-3)

    assert dm.mat.nnz == 4
    assert dm.mat.trace() == pytest.approx(1)

    dm.remove_low_amplitude(1e-1)
    assert dm.mat.nnz == 1
    assert dm.mat.trace() == pytest.approx(1)


def test_divide_fockstate():

    fs = BasicState([2, 0, 0, 1, 4])
    meas, remain = DensityMatrix._divide_fock_state(fs, [0, 2, 4])
    assert meas == BasicState([2, 0, 4])
    assert remain == BasicState([0, 1])


def test_measure_density_matrix():

    plus_state = BasicState([0]) + BasicState([1])
    minus_state = BasicState([0]) - BasicState([1])
    svd = SVDistribution({StateVector(BasicState([0]))*plus_state: 1/3,
                          StateVector(BasicState([1]))*minus_state: 2/3})
    dm = DensityMatrix.from_svd(svd)
    dic = dm.measure([0])
    p0, sub_dm_0 = dic[BasicState([0])]
    p1, sub_dm_1 = dic[BasicState([1])]

    assert p0 == pytest.approx(1/3)
    assert p1 == pytest.approx(2/3)

    assert sub_dm_0.mat.toarray() == pytest.approx(1/2*np.array([[1, 1, 0],
                                                                 [1, 1, 0],
                                                                 [0, 0, 0]]))

    assert sub_dm_1.mat.toarray() == pytest.approx(1/2*np.array([[1, -1],
                                                                 [-1, 1]]))

    sv = BasicState([1, 1, 1, 1]) + BasicState([2, 0, 2, 0]) + BasicState([2, 0, 1, 1])
    equivalent_dm = DensityMatrix.from_svd(sv)

    measurements = sv.measure([0, 1])
    measurements_dm = equivalent_dm.measure([0, 1])

    assert len(measurements) == len(measurements_dm)

    for state in measurements.keys():
        p_sv, rstate = measurements[state]
        p_dm, rdm = measurements_dm[state]

        assert p_sv == pytest.approx(p_dm)

        dm_comparison = DensityMatrix.from_svd(rstate, index=rdm.index)

        assert dm_comparison.mat.toarray() == pytest.approx(rdm.mat.toarray())


def test_photon_loss():

    dm = DensityMatrix.from_svd(BasicState([5]))
    dm.apply_loss(0, .5)

    assert dm.mat.trace() == pytest.approx(1)

    for k in range(6):
        assert dm[BasicState([k]), BasicState([k])] == pytest.approx(math.comb(5, k) * (1/2)**5)

    sv = BasicState([2, 1, 0]) + BasicState([1, 3, 2]) + BasicState([0, 0, 1])
    dm = DensityMatrix.from_svd(sv)
    dm.apply_loss([0, 1], .2)

    assert dm.mat.trace() == pytest.approx(1)

    dm_test = DensityMatrix.from_svd(Source(.5).generate_distribution(BasicState([0, 1, 1])))
    dm_to_test = DensityMatrix.from_svd(Source(.5).generate_distribution(BasicState([1, 1])))

    #  Beam splitter that has a chance 0.75 to make the photon switch its mode (model of a .75 loss probability)
    virtual_circuit = Circuit(3) // (0, Unitary(Matrix([[.5, .866025404],
                                                   [-.866025404, .5]])))

    sim = Simulator(SLOSBackend())
    sim.set_circuit(virtual_circuit)
    out_svd = sim.evolve_density_matrix(dm_test)
    remaining_dms = out_svd.measure(0)

    remaining_dm = sum([prob*dm for (prob, dm) in remaining_dms.values()])

    assert remaining_dm.n_max == 2
    assert remaining_dm.m == 2

    dm_to_test.apply_loss(0, .75)

    assert dm_to_test.n_max == 2
    assert dm_to_test.m == 2

    assert remaining_dm.size == dm_to_test.size
    assert remaining_dm.mat.toarray() == pytest.approx(dm_to_test.mat.toarray())
