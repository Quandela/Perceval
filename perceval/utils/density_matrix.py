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

import random
from copy import copy
from math import comb, sqrt
from typing import Union, Optional, List, Tuple

import numpy as np
from numpy import conj
from scipy.linalg import eigh
from scipy.sparse.linalg import LinearOperator, eigsh
from scipy.sparse import dok_array, csr_array, kron

import exqalibur as xq
from .statevector import StateVector, SVDistribution, BasicState, max_photon_state_iterator, BSSamples
from .density_matrix_utils import array_to_statevector, is_hermitian, sparray

# In all the DensityMatrix Class, there is a compromise between csr_array and dok_array.
# The first one is well suited for matrix-vector product, the other one is easier to construct from scratch

SPARSE_THRESHOLD = 50


class FockBasis(dict):

    def __init__(self, m, n_max):
        super().__init__()
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
    Make the tensor product of 2 Density Matrices \
    :param A, B: two density matrices \
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
    for i, a_state in enumerate(A.inverse_index):
        for j, b_state in enumerate(B.inverse_index):
            index = new_index[a_state*b_state]
            perm[i*B.size+j, index] = 1

    matrix = perm.T @ matrix @ perm

    return DensityMatrix(matrix, new_index, check_hermitian=False)


class DensityMatrix:
    """
    Density operator representing a mixed state. Does not support annotations yet.

    :param mixed_state: 2d-array, SVDistribution, StateVector or Basic State representing a mixed state
    :param index: index of all BasicStates accessible from this mixed states through a unitary evolution
    :param m: optional number of modes if index is not given
    :param n_max: optional maximum number of photon if index is not given
    """
    def __init__(self,
                 mixed_state: Union[np.array, sparray],
                 index: Optional[FockBasis] = None,
                 m: Optional[int] = None,
                 n_max: Optional[int] = None,
                 check_hermitian: bool = True,
                 precision: bool = 1e-6):
        """
        Constructor for the DensityMatrix Class
        """
        # Here the constructor for a matrix
        if not isinstance(mixed_state, (np.ndarray, sparray)):
            raise TypeError(f"Can't construct a density matrix from {type(mixed_state)}")

        if check_hermitian:
            if not is_hermitian(mixed_state):
                raise AssertionError("A density Matrix must be Hermitian")

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

        self.mat = csr_array(mixed_state, dtype=complex)
        self._size = self.mat.shape[0]
        self._m = index.m
        self._n_max = index.n_max
        self.precision = precision

        self.index = dict()
        self.inverse_index = []
        self.set_index(index)  # index construction

    @staticmethod
    def from_svd(svd: Union[SVDistribution, StateVector, BasicState], index: Optional[FockBasis] = None):
        """
        Construct a Density matrix from a SVDistribution.

        :param svd: an SVDistribution object representing the mixed state \
        :param index: the basis in which the density matrix is expressed. Self generated if incorrect \
        :return: the DensityMatrix object corresponding to the SVDistribution given \
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
        matrix = sum([p * (vector @ conj(vector.T)) for vector, p in l])  # This avoids the SparseEfficiencyWarning

        return DensityMatrix(matrix, index, check_hermitian=False)

    def set_index(self, index: dict):
        if index is None:
            k = 0
            for key in max_photon_state_iterator(self._m, self._n_max):
                self.index[key] = k
                self.inverse_index.append(key)
                k += 1
        else:
            if len(index) == self._size:
                self.index = index
                self.inverse_index = [None]*self._size
                for key in index.keys():
                    self.inverse_index[index[key]] = key
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
        Defines the mat_vec function of the Linear operator after the deflation of all the vectors in the vec array.

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
            sv = array_to_statevector(v[:, k], self.inverse_index)
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
                    sv += complex(vec[j][i]) * self.inverse_index[j]
                sv.normalize()
                if sv.m != 0:
                    dic[sv] = val[i]
            else:
                continue
        return SVDistribution(dic)

    def to_svd(self, threshold: Optional[float] = None, batch_size: int = 1):
        """
            Gives back an SVDistribution from the density_matrix

            :param threshold: the threshold when the search for eigen values is stopped.
            :param batch_size: the number of eigen values at each Arnoldi's algorithm iteration.
                Only used if matrix is large enough.
            :return: The SVD object corresponding to the DensityMatrix.
                The StateVector with probability < threshold are removed.
        """
        if threshold is None:
            threshold = self.precision

        if self.size < SPARSE_THRESHOLD:  # if the matrix is small: array eigh method

            return self._to_svd_small(threshold)

        else:  # if the matrix is large: sparse eigsh method
            return self._to_svd_large(threshold, batch_size)

    def __radd__(self, other):
        """
        Method used to be compatible with the sum build-in function of python
        Exists ONLY for this case ONLY!!!
        """

        if other == 0:
            return self
        else:
            raise TypeError("You can only add a Density Matrix to a Density Matrix")

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

        return DensityMatrix(new_mat, new_index, check_hermitian=False)

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

    def remove_low_amplitude(self, threshold: Optional[float] = None):
        """
        Remove the lines and column where the amplitude is below a certain threshold
        """

        if threshold is None:
            threshold = self.precision

        projector = dok_array(self.shape)
        for k in range(self.size):
            if self.mat[k, k] > threshold:
                projector[k, k] = 1

        projector = csr_array(projector)

        self.mat = projector.dot(self.mat).dot(projector)
        self.normalize()

    def normalize(self):
        """
        Normalize the density matrix so that Trace(rho) = 1
        """

        factor = self.mat.trace()

        if abs(factor-1) >= self.precision:
            self.mat = (1/factor)*self.mat

    def sample(self, count: int = 1) -> BSSamples:
        """
        Sample a basic state on the density matrix
        """
        self.normalize()
        samples = random.choices(self.inverse_index, list(self.mat.diagonal()), k=count)
        output = BSSamples()
        for state in samples:
            output.append(state)
        return output

    def measure(self, modes: Union[List[int], int]):
        """
        Makes a measure on a list of modes.
        :param modes: a list of integer for the modes you want to measure
        """
        self.normalize()
        if isinstance(modes, int):
            modes = [modes]

        projectors = self._construct_all_projectors(modes)
        res = dict()  # result fo the form {measured FockState: (remaining density matrix, probability)
        for key_fs, item_list in projectors.items():
            basis = item_list[0]  # FockBasis of possible measurement
            projector = item_list[1]
            prob = item_list[2]
            if prob != 0:
                collapsed_dm = projector @ self.mat @ projector.T  # wave function collapse
                resulting_dm = DensityMatrix(collapsed_dm, basis, check_hermitian=False)
                resulting_dm.normalize()
                res[key_fs] = (prob, resulting_dm)
        return res

    def _construct_projector_one_sample(self, modes, fock_state) -> Tuple[FockBasis, dok_array]:
        """
        Construct the projection operator onto the subspace of some number photons on some mode
        """
        if len(modes) != fock_state.m:
            raise ValueError(f"you can't have {fock_state} in  {len(modes)} number of modes")

        basis = FockBasis(self.m-fock_state.m, self.n_max-fock_state.n)
        projector = dok_array((len(basis), self.size), dtype=float)

        for i, fs in enumerate(self.inverse_index):
            meas_fs, remain_fs = self._divide_fock_state(fs, modes)
            if meas_fs == fock_state:
                projector[basis[remain_fs], i] = 1

        return basis, projector

    def _construct_all_projectors(self, modes: List[int]) -> dict:
        """
        construct all the projectors associated with some modes
        :return: a dictionary with for each measured state a list [fock_basis, projector, probability]
        """
        modes = list(set(modes))
        res = dict()
        for nb_measured_photons in range(self.n_max+1):
            # FockBasis for the remaining density matrices
            remaining_basis = FockBasis(self.m - len(modes), self.n_max - nb_measured_photons)
            for measured_fs in xq.FSArray(len(modes), nb_measured_photons):
                # initialisation of the empty projectors
                res[measured_fs] = [remaining_basis, dok_array((len(remaining_basis), self.size)), 0]

        diag_coefs = self.mat.diagonal()

        for i, fs in enumerate(self.inverse_index):  # construction of the projectors
            prob = abs(diag_coefs[i])
            measured_fs, remaining_fs = self._divide_fock_state(fs, modes)
            remaining_basis = res[measured_fs][0]
            new_basis_idx = remaining_basis[remaining_fs]
            res[measured_fs][1][new_basis_idx, i] = 1
            res[measured_fs][2] += prob

        return res

    @staticmethod
    def _divide_fock_state(fs, modes):
        """
        divide a BasicState into two BasicStates
        """
        measured_fs = []
        remaining_fs = []
        for mode in range(fs.m):
            if mode in modes:
                measured_fs.append(fs[mode])
            else:
                remaining_fs.append(fs[mode])

        return BasicState(measured_fs), BasicState(remaining_fs)

    @staticmethod
    def _get_annihilated_fockstate(fockstate, m, n_photon):
        """
        give the fockstate after loss of n_photon in the mode m
        """

        listed_fs = list(fockstate)
        if listed_fs[m] <= n_photon:
            listed_fs[m] = 0
        else:
            listed_fs[m] -= n_photon
        return BasicState(listed_fs)

    def _construct_loss_operators(self, mode: int, p: float):
        """
        Construct the kraus operators for a loss channel on specified modes
        """
        operators = [dok_array(self.shape, dtype=float) for _ in range(self._n_max+1)]

        # We separate the states depending on the number of lost photons
        # Because there is no more coherence between those states
        for n_photon_loss in range(len(operators)):
            for state, idx in self.index.items():
                n_photon = state[mode]
                if n_photon >= n_photon_loss:
                    result_idx = self.index[self._get_annihilated_fockstate(state, mode, n_photon_loss)]

                    #  Binomial probability, (comes from the simplification of the beam splitter action)
                    operators[n_photon_loss][result_idx, idx] += sqrt(comb(n_photon, n_photon_loss) *
                                                                      (1-p)**(n_photon-n_photon_loss) *
                                                                      p**n_photon_loss)
        return operators

    def apply_loss(self, modes: Union[int, list], prob: float):
        """
        Apply a loss on some mode according to some probability of losing a photon
        Everything works like if the mode was connected to some virtual mode with a beam splitter of reflectivity prob

        :param modes: the mode were you want to simulate a loss
        :param prob: the probability to lose a photon
        """

        if isinstance(modes, int):
            modes = [modes]

        for mode in modes:
            self._apply_loss(mode, prob)

    def _apply_loss(self, mode: int, prob: float):

        matrix_after_loss = csr_array(self.shape, dtype=complex)
        for operator in self._construct_loss_operators(mode, prob):
            matrix_after_loss += operator @ self.mat @ operator.T
        self.mat = matrix_after_loss

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
                                str(self.inverse_index[j]) +
                                self._bra_str(self.inverse_index[i]) +
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
