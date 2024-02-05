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

from abc import ABC, abstractmethod
import copy
import numpy as np
from .tomography import StateTomography
from .tomography_utils import _state_to_dens_matrix, _matrix_to_vector, _get_fixed_basis_ops
from perceval.utils import BasicState
from ._prep_n_meas_circuits import get_preparation_circuit

# from MLEfastprocess import povm_operator, f_data


class TomographyMLE(ABC):
    # Maximum Likelihood Estimation
    def __init__(self, nqubit, operator_processor):
        self._nqubit = nqubit
        self._processor = operator_processor
        self._qst = StateTomography(operator_processor=self._processor)

    @abstractmethod
    def _log_likelihood_func(self, *args, **kwargs) -> float:
        pass

    @abstractmethod
    def _grad_log_likelihood_func(self, *args, **kwargs) -> float:
        pass

    def c_data(self, num_state, i):
        """
        Measures the operator indexed by i after the gate where the input state is indexed by num_states.

        :param num_state:
        :param i:
        :return: list where each element is the probability that we measure one eigenvector of the measurement operator
        """

        l = []
        for j in range(self._nqubit - 1, -1, -1):
            l.append(num_state // (6 ** j))
            num_state = num_state % (6 ** j)

        L = []
        for j in range(self._nqubit - 1, -1, -1):
            L.append(i // (3 ** j))
            i = i % (3 ** j)

        output_distribution = self._qst._compute_probs(prep_state_indices=l, meas_pauli_basis_indices=L)

        # TODO: verify what the following does and how is it different from previous code
        B = []
        for j in range(2**self._nqubit):
            measurement_state = BasicState([])
            for m in range(self._nqubit - 1, -1, -1):
                if (j // (2 ** m)) % 2 == 0:
                    measurement_state *= BasicState([1, 0])
                else:
                    measurement_state *= BasicState([0, 1])
            for m in heralded_modes:
                measurement_state *= BasicState([m[1]])
            if renormalization == None:
                B.append(output_distribution[measurement_state] / (3 ** self._nqubit))
            else:
                B.append(output_distribution[measurement_state] / renormalization / (3 ** self._nqubit))
        return B

    @abstractmethod
    def f_data(self):
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
                M = np.kron(get_preparation_circuit(i // 2).compute_unitary(), M)
                # this compute_matrix seems to have changed to prep_circuit unitary todo: verify and modify
                if i % 2 == 0:
                    v = np.kron(np.array([[1], [0]], dtype='complex_'), v)
                else:
                    v = np.kron(np.array([[0], [1]], dtype='complex_'), v)
            B.append(_state_to_dens_matrix(np.dot(M, v)))
        return B

    @staticmethod
    def _inner_frob(A, B):  # calculate inner product associated to Froebenius norm
        return np.trace(np.dot(np.transpose(np.conjugate(A)), B))

    @staticmethod
    def proj_simplex(eigenvalues):
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
        L = TomographyMLE.proj_simplex(eigenvalues2)
        L.reverse()
        x_0 = _state_to_dens_matrix(np.transpose(np.array([eigenvectors[:, 0]], dtype='complex_')))
        x = (L[0] / np.trace(x_0)) * x_0
        for i in range(1, len(eigenvalues2)):
            x_i = _state_to_dens_matrix(np.transpose(np.array([eigenvectors[:, i]], dtype='complex_')))
            x += (L[i] / np.trace(x_i)) * x_i
        return x


class MLEStateTomography(TomographyMLE):
    def __init__(self, nqubit, operator_processor):
        super().__init__(nqubit, operator_processor)

    def f_data(self):
        """
        Doing a POVM after the gate on the sets of input states and stores it in a dictionary

        :return: dictionary where keys are the indexes of the input state and the
        values are probabilities of each outcome of the POVM
        """
        f = []
        for i in range(3 ** self._nqubit):
            f += self.c_data(prep_state_indices=0, meas_pauli_basis_indices=i)
            # todo : fix this on the basis of Arman's example
        return f

    def _log_likelihood_func(self, f: dict, rho: np.ndarray) -> float:
        """
        Log-likelihood function to minimize
        :param f: dictionary for the data, keys are the input states and the values are probabilities for each outcome
        of the POVM given a certain input (must be called with f_data)
        :param rho: density matrix
        :returns: log-likelihood
        """
        P = self._povm_operator()

        x = 0
        for k in range(len(f)):
            if np.trace(np.dot(rho, P[k])) != 0:
                x -= f[k] * np.log(np.trace(np.dot(rho, P[k])))
        return x

    def _grad_log_likelihood_func(self, f: dict, rho: np.ndarray) -> float:
        """
        Gradient of the log-likelihood function
        :param f: dictionary for the data, keys are the input states and the values are probabilities for each outcome
        of the POVM given a certain input (must be called with f_data)
        :param rho: density matrix
        :returns: gradient of log-likelihood
        """
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
    #     L = TomographyMLE.proj_simplex(eigenvalues2)
    #     L.reverse()
    #     x = L[0] * _state_to_dens_matrix(eigenvectors[:, 0])
    #     for i in range(1, len(eigenvalues2)):
    #         x += L[i] * _state_to_dens_matrix(np.transpose(np.array([eigenvectors[:, i]], dtype='complex_')))
    #     return x

    def APG_state(self, rho_0, f, beta, t, max_it):
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
            rho_proj_i = TomographyMLE._proj(rho - t_i * self._grad_log_likelihood_func(f, rho))
            delta_i = rho_proj_i - rho
            while self._log_likelihood_func(f, rho_proj_i) > self._log_likelihood_func(f, rho) + TomographyMLE._inner_frob(self._grad_log_likelihood_func(f, rho), delta_i) + (
                    1 / (2 * t_i)) * np.linalg.norm(delta_i, ord='fro') ** 2:
                t_i *= beta
                rho_proj_i = TomographyMLE._proj(rho - t_i * self._grad_log_likelihood_func(f, rho))
                delta_i = rho_proj_i - rho
            delta_i_hat = rho_proj_i - rho_proj_i1
            if TomographyMLE._inner_frob(delta_i, delta_i_hat) < 0:
                rho_proj_i, rho, theta = rho_proj_i1, rho_proj_i1, 1
            else:
                theta, rho = (1 + np.sqrt(1 + 4 * theta ** 2)) / 2, rho_proj_i + delta_i_hat * (theta - 1) / (
                            (1 + np.sqrt(1 + 4 * theta ** 2)) / 2)
            if np.abs(self._log_likelihood_func(f, rho_proj_i) - self._log_likelihood_func(f, rho_proj_i1)) < 10 ** (-10):
                break
            rho_proj_i1 = rho_proj_i
        return rho_proj_i


class MLEProcessTomography(TomographyMLE):
    def __init__(self, nqubit, operator_processor):
        super().__init__(nqubit, operator_processor)

    def f_data(self):
        """
        Doing a POVM after the gate on the sets of input states and stores it in a dictionary

        :return: dictionary where keys are the indexes of the input state and the
        values are probabilities of each outcome of the POVM
        """
        f = {}
        for a in range(6 ** self._nqubit):
            f[a] = []
            for b in range(3 ** self._nqubit):
                f[a] += self.c_data(prep_state_indices=a, meas_pauli_basis_indices=b)
        return f

    def _log_likelihood_func(self, f: dict, S: np.ndarray) -> float:
        """
        Log-likelihood function to minimize
        :param f: dictionary for the data, keys are the input states and the values are probabilities for each
        outcome of the POVM given a certain input (must be called with f_data)
        :param S: Choi matrix
        :returns: log-likelihood
        """
        P = self._povm_operator()
        B = self._input_basis()

        x = 0
        for m in range(len(B)):
            for l in range(len(P)):
                pml = 2 ** self._nqubit * np.real(np.trace(np.dot(S, np.kron(np.transpose(B[m]), P[l]))))
                if 0 < pml <= 1:
                    x -= f[m][l] * np.log(pml)
        return x

    def _grad_log_likelihood_func(self, f: dict, S) -> float:
        """
        Gradient of the log-likelihood function
        :param f: dictionary for the data, keys are the input states and the values are
        probabilities for each outcome of the POVM given a certain input (must be called with f_data)
        :param S: Choi matrix
        :returns: gradient of log-likelihood
        """
        P = self._povm_operator()
        B = self._input_basis()

        grad = 0
        for l in range(len(P)):
            for m in range(len(B)):
                pml = 2 ** self._nqubit * np.real(np.trace(np.dot(S, np.kron(np.transpose(B[m]), P[l]))))
                if 0 < pml <= 1:
                    grad -= (f[m][l] / pml) * np.kron(np.transpose(B[m]), P[l])
        return grad

    def APG_process(self, S_0, f, beta=0.5, t=1, max_it=100):
        """
        Accelerated Projected Gradient algorithm for process tomography adapted from https://doi.org/10.48550/arXiv.1609.07881

        :param S_0: initial Choi matrix, usually the identity
        :param f: dictionary for the data, keys are the input states and the values are probabilities for each
        outcome of the POVM given a certain input (must be called with f_data)
        :param beta: parameter to update the learning rate t, to decrease it to make the descent slower when needed
        :param t: initial learning rate
        :param max_it: maximum number of iterations

        :return: Choi matrix maximising likelihood function
        """
        nqubit = int(np.log2(len(S_0)) / 2)
        S, S_proj_i1, theta, t_i = copy.deepcopy(S_0), copy.deepcopy(S_0), 1, t
        for i in range(max_it):
            S_proj_i = TomographyMLE._proj(S - t_i * self._grad_log_likelihood_func(f, S))
            delta_i = S_proj_i - S
            while self._log_likelihood_func(f, S_proj_i) > self._log_likelihood_func(f, S) + TomographyMLE._inner_frob(self._grad_log_likelihood_func(f, S), delta_i) + (
                    1 / (2 * t_i)) * np.linalg.norm(delta_i, ord='fro') ** 2:
                t_i *= beta
                S_proj_i = TomographyMLE._proj(S - t_i * self._grad_log_likelihood_func(f, S))
                delta_i = S_proj_i - S
            delta_i_hat = S_proj_i - S_proj_i1
            if TomographyMLE._inner_frob(delta_i, delta_i_hat) < 0:
                S_proj_i, S, theta = copy.deepcopy(S_proj_i1), copy.deepcopy(S_proj_i1), 1
            else:
                theta, S = (1 + np.sqrt(1 + 4 * theta ** 2)) / 2, S_proj_i + delta_i_hat * (theta - 1) / (
                            (1 + np.sqrt(1 + 4 * theta ** 2)) / 2)
            if np.abs(self._log_likelihood_func(f, S_proj_i) - self._log_likelihood_func(f, S_proj_i1)) < 10 ** (-10):
                break
            S_proj_i1 = copy.deepcopy(S_proj_i)
        return S_proj_i

    def chi_from_choi(self, choi):
        """
        Converts a Choi matrix into a chi matrix
        :param choi: Choi matrix
        :return: chi matrix
        """
        X = np.zeros((len(choi), len(choi)), dtype='complex_')

        for m in range(len(choi)):
            P_m = np.conjugate(np.transpose(_matrix_to_vector(np.transpose(_get_fixed_basis_ops(m, self._nqubit)))))

            for n in range(len(choi)):
                X[m, n] = (1 / 2 ** self._nqubit) * np.linalg.multi_dot(
                    [P_m, choi, _matrix_to_vector(np.transpose(_get_fixed_basis_ops(n, self._nqubit)))])
        return X
