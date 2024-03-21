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

from abc import abstractmethod
import copy
import numpy as np
from .tomography_utils import _state_to_dens_matrix, _matrix_to_vector, _get_fixed_basis_ops, _compute_probs, \
    _generate_pauli_prep_index
from perceval.utils import BasicState
from perceval.components import AProcessor, get_pauli_eigen_state_prep_circ, PauliType, PauliEigenStateType
from ..abstract_algorithm import AAlgorithm


class TomographyMLE(AAlgorithm):
    # Maximum Likelihood Estimation
    def __init__(self, operator_processor: AProcessor, **kwargs):

        super().__init__(processor=operator_processor, **kwargs)
        self._nqubit, odd_modes = divmod(operator_processor.m, 2)
        if odd_modes:
            raise ValueError(
                f"Input processor has an odd mode count ({operator_processor.m}) and thus, is not a logical gate")

        if self._processor.is_remote:
            raise TypeError("Tomography does not support Remote Processor yet")

        self._gate_logical_perf = None

    _LOGICAL0 = BasicState([1, 0])
    _LOGICAL1 = BasicState([0, 1])

    @abstractmethod
    def _log_likelihood_func(self, *args, **kwargs) -> float:
        pass

    @abstractmethod
    def _grad_log_likelihood_func(self, *args, **kwargs) -> float:
        pass

    def _c_data(self, num_state, i):
        """
        Measures the operator indexed by i after the gate where the input state is indexed by num_states.

        :param num_state:
        :param i:
        :return: list where each element is the probability that we measure one eigenvector of the measurement operator
        """

        l1 = []
        for j in range(self._nqubit - 1, -1, -1):
            l1.append(num_state // (6 ** j))
            num_state = num_state % (6 ** j)
        l = [PauliEigenStateType(val) for val in l1]

        L1 = []
        for j in range(self._nqubit - 1, -1, -1):
            L1.append(i // (3 ** j))
            i = i % (3 ** j)
        L = [PauliType(val) for val in L1]
        print('computing probs for prep', l)
        print('computing probs for meas', L)
        output_distribution, self._gate_logical_perf = _compute_probs(self, prep_state_indices=l, meas_pauli_basis_indices=L)

        # TODO: verify what the following does and how is it different from previous code
        B = []
        for j in range(2**self._nqubit):

            measurement_state = BasicState()
            for m in range(self._nqubit - 1, -1, -1):
                if (j // (2 ** m)) % 2 == 0:
                    measurement_state *= self._LOGICAL0
                else:
                    measurement_state *= self._LOGICAL1

            B.append(output_distribution[measurement_state] / (3 ** self._nqubit))
            # if renormalization == None:
            #     B.append(output_distribution[measurement_state] / (3 ** self._nqubit))
            # else:
            #     B.append(output_distribution[measurement_state] / renormalization / (3 ** self._nqubit))
        return B

    @abstractmethod
    def _f_data(self):
        pass

    def _povm_state(self):
        """
        Gives a set of measurement states, so they form a set of informationally complete measurements. For 1 qubit,
        they are : |0>,|1>,|+>,|->,|i+>,|i->,
        which are the eigenvectors of the Pauli operators I,X,Y. For more qubits, the code returns eigenvectors
        of the tensor products of Pauli operators
        """
        d = 2 ** self._nqubit
        P = []
        B = [np.array([[1], [0]], dtype='complex_'), np.array([[0], [1]], dtype='complex_'),
             (1 / np.sqrt(2)) * np.array([[1], [1]], dtype='complex_'),
             (1 / np.sqrt(2)) * np.array([[1], [-1]], dtype='complex_'),
             (1 / np.sqrt(2)) * np.array([[1], [1j]], dtype='complex_'),
             (1 / np.sqrt(2)) * np.array([[1], [-1j]], dtype='complex_')]  #

        for pauli_index in range(3 ** self._nqubit):  # iterates over pauli operators I,X,Y or their tensor products
            pauli_list = []  # saves on each element the pauli index for this qubit

            for j in range(self._nqubit - 1, -1, -1):
                pauli_list.append(pauli_index // (3 ** j))
                pauli_index = pauli_index % (3 ** j)
            X = [np.array([1], dtype='complex_')] * d  # neutral elements for tensor products

            for qubit_index in range(self._nqubit):
                for eigenvector_index in range(d):  # iterates over the eigenvectors of the pauli operator
                    eigenvector_list = []  # saves on each element the eigenvector index for this qubit
                    k1 = eigenvector_index
                    for j1 in range(self._nqubit - 1, -1, -1):
                        eigenvector_list.append(k1 // (2 ** j1))
                        k1 = k1 % (2 ** j1)
                    X[eigenvector_index] = np.kron(X[eigenvector_index],
                                                   B[2 * pauli_list[qubit_index] + eigenvector_list[qubit_index]])
            P += X
        return P

    def _povm_operator(self):
        """
        Gives a POVM set suited for tomography
        """
        B = self._povm_state()
        L = []
        for state in B:
            L.append(_state_to_dens_matrix(state) / (3 ** self._nqubit))
        return L

    def _input_basis(self):  # create a matrix basis from all the tensor products states
        """
        Computes input density matrix basis (same as POVM but not same order)
        """

        B = []
        for j in range(6 ** self._nqubit):
            k = []
            for m in range(self._nqubit - 1, -1, -1):
                k.append(j // (6 ** m))
                j = j % (6 ** m)
            k.reverse()
            M = 1
            v = 1
            for i in k:
                # M = np.kron(compute_matrix(i // 2), M)
                M = np.kron(get_pauli_eigen_state_prep_circ(i // 2).compute_unitary(), M)
                # this compute_matrix seems to have changed to prep_circuit unitary todo: verify and modify
                if i % 2 == 0:
                    v = np.kron(np.array([[1], [0]], dtype='complex_'), v)
                else:
                    v = np.kron(np.array([[0], [1]], dtype='complex_'), v)
            B.append(_state_to_dens_matrix(np.dot(M, v)))
        return B

    @staticmethod
    def _frobenius_inner_product(A, B):  # calculate inner product associated to Frobenius norm
        return np.trace(np.dot(np.transpose(np.conjugate(A)), B))

    @staticmethod
    def _proj_simplex(eigenvalues):
        """
        projects a real eigenspectra sorted in descent order on positive elements with their sum equal to 1
        :param eigenvalues: list of real numbers sorted in descent order
        :return: list of positive numbers with sum equal to 1
        """
        u = 0
        for j in range(1, len(eigenvalues) + 1):
            x = eigenvalues[j - 1] - (1 / j) * (sum(eigenvalues[:j]) - 1)
            if x > 0:
                u = j
        if u == 0:
            w = 0
        else:
            w = (1 / u) * (sum(eigenvalues[:u]) - 1)
        L = []
        for lambda_i in eigenvalues:
            L.append(max(lambda_i - w, 0))
        return L

    @staticmethod
    def _proj(h_matrix):
        """
        Projects a hermitian matrix on the cone of positive semi-definite trace=1 matrices
        :param h_matrix: hermitian matrix
        :return: positive semi-definite trace 1 matrix
        """
        eigenvalues, eigenvectors = np.linalg.eigh(h_matrix)
        eigenvalues2 = list(eigenvalues)
        eigenvalues2.reverse()
        L = TomographyMLE._proj_simplex(eigenvalues2)
        L.reverse()
        x_0 = _state_to_dens_matrix(np.transpose(np.array([eigenvectors[:, 0]], dtype='complex_')))
        x = (L[0] / np.trace(x_0)) * x_0
        for i in range(1, len(eigenvalues2)):
            x_i = _state_to_dens_matrix(np.transpose(np.array([eigenvectors[:, i]], dtype='complex_')))
            x += (L[i] / np.trace(x_i)) * x_i
        return x


class StateTomographyMLE(TomographyMLE):
    def __init__(self, operator_processor):
        super().__init__(operator_processor)

    def _f_data(self):
        """
        Doing a POVM after the gate on the sets of input states and stores it in a dictionary

        :return: dictionary where keys are the indexes of the input state and the
        values are probabilities of each outcome of the POVM
        """
        f = []
        for i in range(3 ** self._nqubit):
            # todo: fix the following
            prep_state_indices = 0 # [PauliType.I]  # 0
            meas_pauli_basis_indices = i
            # _c_data(num_state, i)
            f += self._c_data(prep_state_indices, meas_pauli_basis_indices)
            # todo : fix this on the basis of Arman's example
        return f

    def _log_likelihood_func(self, rho: np.ndarray) -> float:
        """
        Log-likelihood function to minimize
        of the POVM given a certain input (must be called with f_data)
        :param rho: density matrix
        :returns: log-likelihood
        """
        f = self._f_data()  # todo:fix :param f: dictionary for the data, keys are the input states and the
        # values are probabilities for each outcome

        P = self._povm_operator()

        x = 0
        for k in range(len(f)):
            if np.trace(np.dot(rho, P[k])) != 0:
                x -= f[k] * np.log(np.trace(np.dot(rho, P[k])))
        return x

    def _grad_log_likelihood_func(self, rho: np.ndarray) -> float:
        """
        Gradient of the log-likelihood function
        of the POVM given a certain input (must be called with f_data)
        :param rho: density matrix
        :returns: gradient of log-likelihood
        """
        f = self._f_data()  # todo: fix :param f: dictionary for the data, keys are the input states and the
        # values are probabilities for each outcome

        P = self._povm_operator()

        grad = 0
        for k in range(len(f)):
            if np.trace(np.dot(rho, P[k])) != 0:
                grad -= (f[k] / (np.trace(np.dot(rho, P[k])))) * P[k]
        return grad

    # @staticmethod
    # def proj(rho):
    #     """
    #     Projects a hermitian matrix on the cone of positive semi-definite trace 1 matrices
    #     :param rho: hermitian matrix
    #     :return: positive semi-definite trace 1 matrix
    #     """
    #     eigenvalues, eigenvectors = np.linalg.eigh(rho)
    #     eigenvalues2 = list(eigenvalues)
    #     eigenvalues2.reverse()
    #     L = TomographyMLE._proj_simplex(eigenvalues2)
    #     L.reverse()
    #     x = L[0] * _state_to_dens_matrix(eigenvectors[:, 0])
    #     for i in range(1, len(eigenvalues2)):
    #         x += L[i] * _state_to_dens_matrix(np.transpose(np.array([eigenvectors[:, i]], dtype='complex_')))
    #     return x

    def APG_state(self, rho_0, beta, t, max_it):
        """
        Accelerated Projected Gradient algorithm for state tomography from https://doi.org/10.48550/arXiv.1609.07881

        :param rho_0: initial density matrix, usually the identity
        :param f: dictionary for the data, keys are the input states and the values are probabilities for each outcome
        of the POVM given a certain input (must be called with f_data)
        :param beta: parameter to update the learning rate t, to decrease it to make the descent slower when needed
        :param t: initial learning rate
        :param max_it: maximum number of iterations

        :return: Density matrix maximising likelihood function
        """
        rho, rho_proj_i1, theta, t_i = rho_0, rho_0, 1, t
        for i in range(max_it):
            rho_proj_i = TomographyMLE._proj(rho - t_i * self._grad_log_likelihood_func(rho))
            delta_i = rho_proj_i - rho
            while self._log_likelihood_func(rho_proj_i) > self._log_likelihood_func(rho) + TomographyMLE._frobenius_inner_product(self._grad_log_likelihood_func(rho), delta_i) + (
                    1 / (2 * t_i)) * np.linalg.norm(delta_i, ord='fro') ** 2:
                t_i *= beta
                rho_proj_i = TomographyMLE._proj(rho - t_i * self._grad_log_likelihood_func(rho))
                delta_i = rho_proj_i - rho
            delta_i_hat = rho_proj_i - rho_proj_i1
            if TomographyMLE._frobenius_inner_product(delta_i, delta_i_hat) < 0:
                rho_proj_i, rho, theta = rho_proj_i1, rho_proj_i1, 1
            else:
                theta, rho = (1 + np.sqrt(1 + 4 * theta ** 2)) / 2, rho_proj_i + delta_i_hat * (theta - 1) / (
                            (1 + np.sqrt(1 + 4 * theta ** 2)) / 2)
            if np.abs(self._log_likelihood_func(rho_proj_i) - self._log_likelihood_func(rho_proj_i1)) < 10 ** (-10):
                break
            rho_proj_i1 = rho_proj_i
        return rho_proj_i


class ProcessTomographyMLE(TomographyMLE):
    def __init__(self, operator_processor):
        super().__init__(operator_processor)

    def _f_data(self):
        """
        Doing a POVM after the gate on the sets of input states and stores it in a dictionary

        :return: dictionary where keys are the indexes of the input state and the
        values are probabilities of each outcome of the POVM
        """
        # todo: fix indices based on the following note
        # measurement is always on 3 Paulitype I, x, Y : Z is moved away
        # prep has 6 options -> 4 pauli and 2 other are some combo
        # of that itself -> find which and decide how to implement
        f = {}
        for a in range(6 ** self._nqubit):
            f[a] = []
            for b in range(3 ** self._nqubit):
                # _c_data(num_state, i)
                # todo: fix indices
                prep_state_indices = a
                meas_pauli_basis_indices = b
                f[a] += self._c_data(num_state=prep_state_indices, i=meas_pauli_basis_indices)
        return f

    def _log_likelihood_func(self, S: np.ndarray) -> float:
        """
        Log-likelihood function to minimize
        outcome of the POVM given a certain input (must be called with _f_data)
        :param S: Choi matrix
        :returns: log-likelihood
        """
        f = self._f_data()  # todo: fix :param f: dictionary for the data, keys are the input states and the values are probabilities for each

        P = self._povm_operator()
        B = self._input_basis()

        x = 0
        for m in range(len(B)):
            for l in range(len(P)):
                pml = 2 ** self._nqubit * np.real(np.trace(np.dot(S, np.kron(np.transpose(B[m]), P[l]))))
                if 0 < pml <= 1:
                    x -= f[m][l] * np.log(pml)
        return x

    def _grad_log_likelihood_func(self, S) -> float:
        """
        Gradient of the log-likelihood function
        probabilities for each outcome of the POVM given a certain input (must be called with _f_data)
        :param S: Choi matrix
        :returns: gradient of log-likelihood
        """
        f = self._f_data()  # todo: fix :param f: dictionary for the data, keys are the input states and the values are

        P = self._povm_operator()
        B = self._input_basis()

        grad = 0
        for l in range(len(P)):
            for m in range(len(B)):
                pml = 2 ** self._nqubit * np.real(np.trace(np.dot(S, np.kron(np.transpose(B[m]), P[l]))))
                if 0 < pml <= 1:
                    grad -= (f[m][l] / pml) * np.kron(np.transpose(B[m]), P[l])
        return grad

    # @staticmethod
    # def proj(S):
    #     """
    #     Projects a hermitian matrix on the cone of positive semi-definite trace 1 matrices
    #     :param S: hermitian matrix
    #     :return: positive semi-definite trace 1 matrix
    #     """
    #     eigenvalues, eigenvectors = np.linalg.eigh(S)
    #     eigenvalues2 = list(eigenvalues)
    #     eigenvalues2.reverse()
    #     L = _proj_simplex(eigenvalues2)
    #     L.reverse()
    #     x_0 = state_to_dens_matrix(np.transpose(np.array([eigenvectors[:, 0]], dtype='complex_')))
    #     x = (L[0] / np.trace(x_0)) * x_0
    #     for i in range(1, len(eigenvalues2)):
    #         x_i = state_to_dens_matrix(np.transpose(np.array([eigenvectors[:, i]], dtype='complex_')))
    #         x += (L[i] / np.trace(x_i)) * x_i
    #     return x

    def _choi_matrix(self, S_0, beta, t, max_it):
        """
        Computes the Choi matrix for the process under study using the Accelerated Projected Gradient algorithm
         for process tomography adapted from https://doi.org/10.48550/arXiv.1609.07881

        :param S_0: initial Choi matrix, usually the identity
        :param beta: parameter to update the learning rate t, to decrease it to make the descent slower when needed
        :param t: initial learning rate
        :param max_it: maximum number of iterations

        :return: Choi matrix maximising likelihood function
        """
        S, S_proj_i1, theta, t_i = copy.deepcopy(S_0), copy.deepcopy(S_0), 1, t
        for i in range(max_it):
            S_proj_i = TomographyMLE._proj(S - t_i * self._grad_log_likelihood_func(S))
            delta_i = S_proj_i - S
            while self._log_likelihood_func(S_proj_i) > self._log_likelihood_func(S) + TomographyMLE._frobenius_inner_product(self._grad_log_likelihood_func(S), delta_i) + (
                    1 / (2 * t_i)) * np.linalg.norm(delta_i, ord='fro') ** 2:
                t_i *= beta
                S_proj_i = TomographyMLE._proj(S - t_i * self._grad_log_likelihood_func(S))
                delta_i = S_proj_i - S
            delta_i_hat = S_proj_i - S_proj_i1
            if TomographyMLE._frobenius_inner_product(delta_i, delta_i_hat) < 0:
                S_proj_i, S, theta = copy.deepcopy(S_proj_i1), copy.deepcopy(S_proj_i1), 1
            else:
                theta, S = (1 + np.sqrt(1 + 4 * theta ** 2)) / 2, S_proj_i + delta_i_hat * (theta - 1) / (
                            (1 + np.sqrt(1 + 4 * theta ** 2)) / 2)
            if np.abs(self._log_likelihood_func(S_proj_i) - self._log_likelihood_func(S_proj_i1)) < 10 ** (-10):
                break
            S_proj_i1 = copy.deepcopy(S_proj_i)
        return S_proj_i

    def chi_matrix(self, S_0, beta=0.5, t=1, max_it=100):
        """
        Converts the Choi matrix into a chi matrix
        # todo: fix params
        :param S_0: initial Choi matrix, usually the identity
        :param beta: parameter to update the learning rate t, to decrease it to make the descent slower when needed
        :param t: initial learning rate
        :param max_it: maximum number of iterations

        :return: chi matrix
        """
        choi = self._choi_matrix(S_0, self._f_data(), beta, t, max_it)
        X = np.zeros((len(choi), len(choi)), dtype='complex_')

        for m in range(len(choi)):
            P_m = np.conjugate(np.transpose(_matrix_to_vector(np.transpose(_get_fixed_basis_ops(m, self._nqubit)))))

            for n in range(len(choi)):
                X[m, n] = (1 / 2 ** self._nqubit) * np.linalg.multi_dot(
                    [P_m, choi, _matrix_to_vector(np.transpose(_get_fixed_basis_ops(n, self._nqubit)))])
        return X
