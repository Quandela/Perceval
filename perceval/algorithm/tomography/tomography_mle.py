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
    _generate_pauli_prep_index, _generate_pauli_index
from perceval.utils import BasicState
from perceval.components import AProcessor, get_pauli_gate, PauliType, PauliEigenStateType, get_pauli_basis_measurement_circuit
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
    _FLAG = 0

    @abstractmethod
    def _log_likelihood_func(self, *args, **kwargs) -> float:
        pass

    @abstractmethod
    def _grad_log_likelihood_func(self, *args, **kwargs) -> float:
        pass

    def _c_data(self, prep_state_indices, state_meas_indices):
        """
        Measures the operator indexed by i after the gate where the input state is indexed by num_states.

        :param prep_state_indices: list of indices for state preparation at each qubit
        :param state_meas_indices: list of indices for state measurement at each qubit
        :return: list where each element is the probability that we measure one eigenvector of the measurement operator
        """

        output_distribution, self._gate_logical_perf = _compute_probs(self, prep_state_indices, state_meas_indices, denormalize=False)
        # print('prep', prep_state_indices)
        # print('meas', state_meas_indices)
        # print(output_distribution)
        # #for key in output_distribution:  # Renormalize output state distribution for MLE
        #    output_distribution[key] /= self._gate_logical_perf

        B = []
        for j in range(2**self._nqubit):
            measurement_state = BasicState()
            for m in range(self._nqubit - 1, -1, -1):
                if (j // (2 ** m)) % 2 == 0:
                    measurement_state *= self._LOGICAL0
                else:
                    measurement_state *= self._LOGICAL1

            B.append(output_distribution[measurement_state] / (3 ** self._nqubit))
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

    def _compute_matrix(self, j):
        if j == 0:
            return np.eye((2), dtype='complex_')
        if j == 1:
            return (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype='complex_')
        if j == 2:
            return (1 / np.sqrt(2)) * np.array([[1, 1], [1j, -1j]], dtype='complex_')

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
                # M = np.kron(get_pauli_gate(PauliType(i // 2)), M)
                #M = np.kron(get_pauli_basis_measurement_circuit(PauliType(i // 2)).compute_unitary(), M)
                M = np.kron(self._compute_matrix(i // 2), M)
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
        #print('previous L', LL)
        #L = [1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
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
        measurement_indices = _generate_pauli_index(self._nqubit)
        preparation_indices = [PauliEigenStateType.Zm]  * self._nqubit
        # Input Preparation fixed to |0> for state tomography

        for val in measurement_indices:
            if PauliType.Z in val:
                continue
            f += self._c_data(preparation_indices, state_meas_indices=val)

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
        self._f = self._f_data()

    def _f_data(self):
        """
        Doing a POVM after the gate on the sets of input states and stores it in a dictionary

        :return: dictionary where keys are the indexes of the input state and the
        values are probabilities of each outcome of the POVM
        """
        # measurement is always on 3 Paulitype I, X, Y : Z is moved away
        # prep has 6 options -> 4 pauli and 2 other are some combo
        # of that itself -> find which and decide how to implement
        f = {}
        preparation_states = _generate_pauli_prep_index(self._nqubit)
        measurement_states = _generate_pauli_index(self._nqubit)

        for index, value in enumerate(preparation_states):
            f[index] = []
            for meas_indices in measurement_states:
                if PauliType.Z in meas_indices:
                    continue
                f[index] += self._c_data(prep_state_indices=value, state_meas_indices=meas_indices)

        return f

    def _log_likelihood_func(self, S: np.ndarray) -> float:
        """
        Log-likelihood function to minimize
        outcome of the POVM given a certain input (must be called with _f_data)
        :param S: Choi matrix
        :returns: log-likelihood
        """
        f = self._f
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
        f = self._f
        P = self._povm_operator()
        B = self._input_basis()

        grad = 0 # a numpy array, not a number
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
        log_f_S = self._log_likelihood_func(S)

        grad_log_f_S = self._grad_log_likelihood_func(S)

       #  grad_log_f_S = np.array([[-1.06699579e+01+0.00000000e+00j, -1.34946442e-02-1.89952221e-16j,
       #   2.78423118e-16-2.34187669e-17j, -3.20923843e-17-6.09416250e-17j,
       #  -4.59966901e-03-8.06646416e-17j, -1.77885760e+00-7.17326263e-17j,
       #   4.33680869e-18+4.59520668e-18j,  2.34368723e-18-1.30104261e-17j,
       #   3.46944695e-18+9.54097912e-18j,  4.59701721e-17+3.12250226e-17j,
       #  -1.11083811e-04+3.12250226e-17j, -8.88872163e-01-1.83013327e-16j,
       #   6.93889390e-18+1.44867516e-17j, -1.44867516e-17+8.67361738e-19j,
       #  -8.88872168e-01-9.02056208e-17j,  1.53512035e-03-4.22423272e-17j],
       # [-1.34946442e-02+1.89952221e-16j, -9.33501491e+00+0.00000000e+00j,
       #   3.20923843e-17+5.00806774e-17j,  1.67400815e-16-1.34441069e-16j,
       #  -4.43914614e-01+1.65407694e-16j, -4.59966901e-03-2.51534904e-17j,
       #  -3.35687098e-17-1.47451495e-17j,  4.33680869e-18+4.59520668e-18j,
       #   6.50521303e-17-5.89805982e-17j, -1.04083409e-17+9.54097912e-18j,
       #  -4.44428404e-01+4.42354486e-17j, -3.17752979e-03+3.12250226e-17j,
       #  -1.32688241e-17-9.54097912e-18j,  1.73472348e-18-1.32688241e-17j,
       #  -1.53132562e-03+2.83645394e-17j, -4.44428409e-01-4.85722573e-17j],
       # [ 2.78423118e-16+2.34187669e-17j,  3.20923843e-17-5.00806774e-17j,
       #  -7.99833670e+00+0.00000000e+00j, -6.77502544e-04-7.89299182e-17j,
       #   9.54097912e-18-4.59520668e-18j, -2.31603689e-17+1.30104261e-17j,
       #   4.59966901e-03-2.51534904e-17j, -8.89437000e-01-9.94882019e-17j,
       #   2.60208521e-18+2.42861287e-17j, -6.59194921e-17+7.19910243e-17j,
       #  -2.42861287e-17-1.82145965e-17j,  4.59701721e-17+3.46944695e-18j,
       #  -2.21435641e-16+5.20417043e-17j, -2.60208521e-18+2.83645394e-17j,
       #   6.93889390e-18+1.44867516e-17j, -1.44867516e-17+8.67361738e-19j],
       # [-3.20923843e-17+6.09416250e-17j,  1.67400815e-16+1.34441069e-16j,
       #  -6.77502544e-04+7.89299182e-17j, -7.99669049e+00+0.00000000e+00j,
       #  -2.88813353e-17+1.47451495e-17j,  9.54097912e-18-4.59520668e-18j,
       #  -8.87790791e-01+1.23774331e-16j,  4.59966901e-03+1.64798730e-17j,
       #   3.81639165e-17+6.67868538e-17j,  2.60208521e-18+2.42861287e-17j,
       #   6.50521303e-17-5.89805982e-17j, -3.81639165e-17-1.82145965e-17j,
       #   8.67361738e-19-6.08963750e-19j,  6.08963750e-19+4.16333634e-17j,
       #  -1.32688241e-17-9.54097912e-18j,  1.73472348e-18-1.32688241e-17j],
       # [-4.59966901e-03+8.06646416e-17j, -4.43914614e-01-1.65407694e-16j,
       #   9.54097912e-18+4.59520668e-18j, -2.88813353e-17-1.47451495e-17j,
       #  -9.33501491e+00+0.00000000e+00j, -1.34946442e-02-7.54604712e-17j,
       #   1.63931368e-16-1.30971622e-16j,  1.99493200e-17-6.09416250e-17j,
       #  -1.21430643e-17-2.86048320e-18j,  9.79937711e-18+2.60208521e-18j,
       #  -4.44428409e-01-4.68375339e-17j, -1.53132562e-03-1.10173046e-17j,
       #   1.04083409e-17+1.30104261e-17j,  4.25007252e-17+1.38777878e-17j,
       #  -3.17752979e-03+2.42861287e-17j, -4.44428404e-01-1.44849410e-16j],
       # [-1.77885760e+00+7.17326263e-17j, -4.59966901e-03+2.51534904e-17j,
       #  -2.31603689e-17-1.30104261e-17j,  9.54097912e-18+4.59520668e-18j,
       #  -1.34946442e-02+7.54604712e-17j, -1.06699579e+01+0.00000000e+00j,
       #   3.55618313e-17+5.00806774e-17j,  2.74953671e-16-1.99493200e-17j,
       #  -9.79937711e-18+2.60208521e-18j,  6.93889390e-18+1.10173046e-17j,
       #   1.53512035e-03-1.67382710e-17j, -8.88872168e-01-8.84708973e-17j,
       #   3.38271078e-17-4.16333634e-17j,  2.42861287e-17+2.68882139e-17j,
       #  -8.88872163e-01-1.47451495e-17j, -1.11083811e-04+1.04083409e-17j],
       # [ 4.33680869e-18-4.59520668e-18j, -3.35687098e-17+1.47451495e-17j,
       #   4.59966901e-03+2.51534904e-17j, -8.87790791e-01-1.23774331e-16j,
       #   1.63931368e-16+1.30971622e-16j,  3.55618313e-17-5.00806774e-17j,
       #  -7.99669049e+00+0.00000000e+00j, -6.77502544e-04+9.10729825e-17j,
       #   2.48950924e-17+5.72458747e-17j, -1.12757026e-17+4.07841070e-18j,
       #  -1.21430643e-17-2.86048320e-18j,  9.79937711e-18+2.60208521e-18j,
       #   3.38271078e-17+5.89805982e-17j,  1.24900090e-16+3.38271078e-17j,
       #   1.04083409e-17+2.68882139e-17j,  4.25007252e-17-4.16333634e-17j],
       # [ 2.34368723e-18+1.30104261e-17j,  4.33680869e-18-4.59520668e-18j,
       #  -8.89437000e-01+9.94882019e-17j,  4.59966901e-03-1.64798730e-17j,
       #   1.99493200e-17+6.09416250e-17j,  2.74953671e-16+1.99493200e-17j,
       #  -6.77502544e-04-9.10729825e-17j, -7.99833670e+00+0.00000000e+00j,
       #   2.60208521e-18+9.79937711e-18j,  8.04062436e-17+6.76542156e-17j,
       #  -9.79937711e-18+2.60208521e-18j,  6.93889390e-18+1.10173046e-17j,
       #   1.24900090e-16+9.80118764e-17j,  1.99493200e-17+7.28583860e-17j,
       #   3.38271078e-17-4.16333634e-17j,  2.42861287e-17-8.67361738e-19j],
       # [ 3.46944695e-18-9.54097912e-18j,  6.50521303e-17+5.89805982e-17j,
       #   2.60208521e-18-2.42861287e-17j,  3.81639165e-17-6.67868538e-17j,
       #  -1.21430643e-17+2.86048320e-18j, -9.79937711e-18-2.60208521e-18j,
       #   2.48950924e-17-5.72458747e-17j,  2.60208521e-18-9.79937711e-18j,
       #  -8.00330950e+00+0.00000000e+00j, -3.74531919e-03+3.20923843e-17j,
       #   3.06178694e-16-2.73218947e-16j, -3.20923843e-17-3.31860494e-17j,
       #  -4.59966901e-03-9.54097912e-17j, -8.89986982e-01-7.86715202e-17j,
       #  -8.67361738e-18+4.59520668e-18j,  2.34368723e-18-8.67361738e-18j],
       # [ 4.59701721e-17-3.12250226e-17j, -1.04083409e-17-9.54097912e-18j,
       #  -6.59194921e-17-7.19910243e-17j,  2.60208521e-18-2.42861287e-17j,
       #   9.79937711e-18-2.60208521e-18j,  6.93889390e-18-1.10173046e-17j,
       #  -1.12757026e-17-4.07841070e-18j,  8.04062436e-17-6.76542156e-17j,
       #  -3.74531919e-03-3.20923843e-17j, -8.00166330e+00+0.00000000e+00j,
       #   3.20923843e-17+7.78362530e-17j,  3.06178694e-16-2.73218947e-16j,
       #  -8.88340783e-01+1.06427096e-16j, -4.59966901e-03-1.21430643e-17j,
       #  -3.35687098e-17-5.20417043e-18j,  1.90819582e-17+4.59520668e-18j],
       # [-1.11083811e-04-3.12250226e-17j, -4.44428404e-01-4.42354486e-17j,
       #  -2.42861287e-17+1.82145965e-17j,  6.50521303e-17+5.89805982e-17j,
       #  -4.44428409e-01+4.68375339e-17j,  1.53512035e-03+1.67382710e-17j,
       #  -1.21430643e-17+2.86048320e-18j, -9.79937711e-18-2.60208521e-18j,
       #   3.06178694e-16+2.73218947e-16j,  3.20923843e-17-7.78362530e-17j,
       #  -9.33168451e+00+0.00000000e+00j,  7.97184851e-03-3.00974523e-16j,
       #   2.25514052e-17-4.59520668e-18j, -2.31603689e-17+8.67361738e-18j,
       #   4.59966901e-03-1.90819582e-17j, -4.45007036e-01-9.94882019e-17j],
       # [-8.88872163e-01+1.83013327e-16j, -3.17752979e-03-3.12250226e-17j,
       #   4.59701721e-17-3.46944695e-18j, -3.81639165e-17+1.82145965e-17j,
       #  -1.53132562e-03+1.10173046e-17j, -8.88872168e-01+8.84708973e-17j,
       #   9.79937711e-18-2.60208521e-18j,  6.93889390e-18-1.10173046e-17j,
       #  -3.20923843e-17+3.31860494e-17j,  3.06178694e-16+2.73218947e-16j,
       #   7.97184851e-03+3.00974523e-16j, -1.06633427e+01+0.00000000e+00j,
       #  -2.88813353e-17+5.20417043e-18j, -5.20417043e-18-4.59520668e-18j,
       #  -1.77666520e+00+9.94882019e-17j,  4.59966901e-03+6.41847686e-17j],
       # [ 6.93889390e-18-1.44867516e-17j, -1.32688241e-17+9.54097912e-18j,
       #  -2.21435641e-16-5.20417043e-17j,  8.67361738e-19+6.08963750e-19j,
       #   1.04083409e-17-1.30104261e-17j,  3.38271078e-17+4.16333634e-17j,
       #   3.38271078e-17-5.89805982e-17j,  1.24900090e-16-9.80118764e-17j,
       #  -4.59966901e-03+9.54097912e-17j, -8.88340783e-01-1.06427096e-16j,
       #   2.25514052e-17+4.59520668e-18j, -2.88813353e-17-5.20417043e-18j,
       #  -8.00166330e+00+0.00000000e+00j, -3.74531919e-03+3.55618313e-17j,
       #   3.02709247e-16-1.58727198e-16j, -3.55618313e-17-3.31860494e-17j],
       # [-1.44867516e-17-8.67361738e-19j,  1.73472348e-18+1.32688241e-17j,
       #  -2.60208521e-18-2.83645394e-17j,  6.08963750e-19-4.16333634e-17j,
       #   4.25007252e-17-1.38777878e-17j,  2.42861287e-17-2.68882139e-17j,
       #   1.24900090e-16-3.38271078e-17j,  1.99493200e-17-7.28583860e-17j,
       #  -8.89986982e-01+7.86715202e-17j, -4.59966901e-03+1.21430643e-17j,
       #  -2.31603689e-17-8.67361738e-18j, -5.20417043e-18+4.59520668e-18j,
       #  -3.74531919e-03-3.55618313e-17j, -8.00330950e+00+0.00000000e+00j,
       #   3.55618313e-17+7.78362530e-17j,  3.02709247e-16-1.58727198e-16j],
       # [-8.88872168e-01+9.02056208e-17j, -1.53132562e-03-2.83645394e-17j,
       #   6.93889390e-18-1.44867516e-17j, -1.32688241e-17+9.54097912e-18j,
       #  -3.17752979e-03-2.42861287e-17j, -8.88872163e-01+1.47451495e-17j,
       #   1.04083409e-17-2.68882139e-17j,  3.38271078e-17+4.16333634e-17j,
       #  -8.67361738e-18-4.59520668e-18j, -3.35687098e-17+5.20417043e-18j,
       #   4.59966901e-03+1.90819582e-17j, -1.77666520e+00-9.94882019e-17j,
       #   3.02709247e-16+1.58727198e-16j,  3.55618313e-17-7.78362530e-17j,
       #  -1.06633427e+01+0.00000000e+00j,  7.97184851e-03-2.41993925e-16j],
       # [ 1.53512035e-03+4.22423272e-17j, -4.44428409e-01+4.85722573e-17j,
       #  -1.44867516e-17-8.67361738e-19j,  1.73472348e-18+1.32688241e-17j,
       #  -4.44428404e-01+1.44849410e-16j, -1.11083811e-04-1.04083409e-17j,
       #   4.25007252e-17+4.16333634e-17j,  2.42861287e-17+8.67361738e-19j,
       #   2.34368723e-18+8.67361738e-18j,  1.90819582e-17-4.59520668e-18j,
       #  -4.45007036e-01+9.94882019e-17j,  4.59966901e-03-6.41847686e-17j,
       #  -3.55618313e-17+3.31860494e-17j,  3.02709247e-16+1.58727198e-16j,
       #   7.97184851e-03+2.41993925e-16j, -9.33168451e+00+0.00000000e+00j]])

        # print("CHECKING GRAD")
        # print(np.all(grad_log_f_S == grad_log_f_S_raksha))
        # import matplotlib.pyplot as plt
        # plt.imshow(np.abs(grad_log_f_S_raksha-grad_log_f_S))
        # plt.colorbar()
        # plt.show()

        for i in range(max_it):
            print('iteration', i)

            S_proj_i = TomographyMLE._proj(S - t_i * grad_log_f_S)

            # S_proj_i = np.array([[2.50618600e-01 + 0.00000000e+00j, 1.17503838e-03 + 1.13184367e-17j,
            #                       -1.07648659e-16 + 2.24116591e-18j, 3.48436389e-18 + 1.06923457e-18j,
            #                       1.17503838e-03 + 1.60957201e-18j, 2.50618600e-01 + 1.47894322e-17j,
            #                       -1.84940908e-18 - 8.34514998e-19j, -2.58368931e-17 + 4.27317844e-19j,
            #                       1.69415215e-18 - 1.32900372e-18j, -1.51988854e-17 - 1.39694931e-17j,
            #                       -5.52301217e-04 - 1.50603299e-17j, 2.49995863e-01 + 4.23860698e-17j,
            #                       -1.48178676e-17 - 7.48730420e-18j, -1.37651100e-18 + 2.00486664e-18j,
            #                       2.49995863e-01 + 2.52753605e-17j, -5.52301217e-04 + 1.04795834e-17j],
            #                      [1.17503838e-03 - 1.13184367e-17j, 5.50922872e-06 + 0.00000000e+00j,
            #                       -5.04716352e-19 + 1.05078233e-20j, 1.63366218e-20 + 5.01316204e-21j,
            #                       5.50922872e-06 - 4.55205184e-20j, 1.17503838e-03 - 1.12490957e-17j,
            #                       -8.67105094e-21 - 3.91266709e-21j, -1.21137621e-19 + 2.00350200e-21j,
            #                       7.94312070e-21 - 6.23110325e-21j, -7.12607671e-20 - 6.54966968e-20j,
            #                       -2.58949306e-06 - 4.56681157e-20j, 1.17211864e-03 - 1.10915833e-17j,
            #                       -6.94743451e-20 - 3.51046162e-20j, -6.45384361e-21 + 9.39992179e-21j,
            #                       1.17211864e-03 - 1.11718078e-17j, -2.58949306e-06 + 7.40770999e-20j],
            #                      [-1.07648659e-16 - 2.24116591e-18j, -5.04716352e-19 - 1.05078233e-20j,
            #                       4.62585642e-32 + 0.00000000e+00j, -1.48708343e-33 - 4.90429301e-34j,
            #                       -5.04716352e-19 - 1.05078233e-20j, -1.07648659e-16 - 2.24116591e-18j,
            #                       7.86917336e-34 + 3.74989139e-34j, 1.11015885e-32 + 4.75007490e-35j,
            #                       -7.39576888e-34 + 5.55699345e-34j, 6.40348195e-33 + 6.13625811e-33j,
            #                       2.37230937e-19 + 4.93897365e-21j, -1.07381174e-16 - 2.23559707e-18j,
            #                       6.29778989e-33 + 3.34854458e-33j, 6.09183842e-34 - 8.48844479e-34j,
            #                       -1.07381174e-16 - 2.23559707e-18j, 2.37230937e-19 + 4.93897365e-21j],
            #                      [3.48436389e-18 - 1.06923457e-18j, 1.63366218e-20 - 5.01316204e-21j,
            #                       -1.48708343e-33 + 4.90429301e-34j, 5.30050614e-35 + 0.00000000e+00j,
            #                       1.63366218e-20 - 5.01316204e-21j, 3.48436389e-18 - 1.06923457e-18j,
            #                       -2.92727934e-35 - 3.71202217e-36j, -3.57388614e-34 + 1.16171066e-34j,
            #                       1.78838516e-35 - 2.57051097e-35j, -2.70910110e-34 - 1.29374370e-34j,
            #                       -7.67867356e-21 + 2.35632772e-21j, 3.47570594e-18 - 1.06657773e-18j,
            #                       -2.37957307e-34 - 4.08777166e-35j, -1.05841805e-35 + 3.37464899e-35j,
            #                       3.47570594e-18 - 1.06657773e-18j, -7.67867356e-21 + 2.35632772e-21j],
            #                      [1.17503838e-03 - 1.60957201e-18j, 5.50922872e-06 + 4.55205184e-20j,
            #                       -5.04716352e-19 + 1.05078233e-20j, 1.63366218e-20 + 5.01316204e-21j,
            #                       5.50922872e-06 + 0.00000000e+00j, 1.17503838e-03 - 1.54023098e-18j,
            #                       -8.67105094e-21 - 3.91266709e-21j, -1.21137621e-19 + 2.00350200e-21j,
            #                       7.94312070e-21 - 6.23110325e-21j, -7.12607671e-20 - 6.54966968e-20j,
            #                       -2.58949306e-06 - 6.70640448e-20j, 1.17211864e-03 - 1.40684324e-18j,
            #                       -6.94743451e-20 - 3.51046162e-20j, -6.45384361e-21 + 9.39992179e-21j,
            #                       1.17211864e-03 - 1.48706770e-18j, -2.58949306e-06 + 5.26811708e-20j],
            #                      [2.50618600e-01 - 1.47894322e-17j, 1.17503838e-03 + 1.12490957e-17j,
            #                       -1.07648659e-16 + 2.24116591e-18j, 3.48436389e-18 + 1.06923457e-18j,
            #                       1.17503838e-03 + 1.54023098e-18j, 2.50618600e-01 + 0.00000000e+00j,
            #                       -1.84940908e-18 - 8.34514998e-19j, -2.58368931e-17 + 4.27317844e-19j,
            #                       1.69415215e-18 - 1.32900372e-18j, -1.51988854e-17 - 1.39694931e-17j,
            #                       -5.52301217e-04 - 1.50277377e-17j, 2.49995863e-01 + 2.76333864e-17j,
            #                       -1.48178676e-17 - 7.48730420e-18j, -1.37651100e-18 + 2.00486664e-18j,
            #                       2.49995863e-01 + 1.05226771e-17j, -5.52301217e-04 + 1.05121757e-17j],
            #                      [-1.84940908e-18 + 8.34514998e-19j, -8.67105094e-21 + 3.91266709e-21j,
            #                       7.86917336e-34 - 3.74989139e-34j, -2.92727934e-35 + 3.71202217e-36j,
            #                       -8.67105094e-21 + 3.91266709e-21j, -1.84940908e-18 + 8.34514998e-19j,
            #                       1.64262718e-35 + 0.00000000e+00j, 1.89237277e-34 - 8.91855602e-35j,
            #                       -8.07644298e-36 + 1.54484421e-35j, 1.58674209e-34 + 5.24765899e-35j,
            #                       4.07563879e-21 - 1.83906402e-21j, -1.84481367e-18 + 8.32441395e-19j,
            #                       1.34278009e-34 + 5.91079689e-36j, 3.48194694e-36 - 1.93782011e-35j,
            #                       -1.84481367e-18 + 8.32441395e-19j, 4.07563879e-21 - 1.83906402e-21j],
            #                      [-2.58368931e-17 - 4.27317844e-19j, -1.21137621e-19 - 2.00350200e-21j,
            #                       1.11015885e-32 - 4.75007490e-35j, -3.57388614e-34 - 1.16171066e-34j,
            #                       -1.21137621e-19 - 2.00350200e-21j, -2.58368931e-17 - 4.27317844e-19j,
            #                       1.89237277e-34 + 8.91855602e-35j, 2.66431798e-33 + 0.00000000e+00j,
            #                       -1.76920367e-34 + 1.34121671e-34j, 1.54307208e-33 + 1.46606458e-33j,
            #                       5.69381023e-20 + 9.41702512e-22j, -2.57726935e-17 - 4.26256044e-19j,
            #                       1.51484448e-33 + 7.97150000e-34j, 1.45326335e-34 - 2.04339650e-34j,
            #                       -2.57726935e-17 - 4.26256044e-19j, 5.69381023e-20 + 9.41702512e-22j],
            #                      [1.69415215e-18 + 1.32900372e-18j, 7.94312070e-21 + 6.23110325e-21j,
            #                       -7.39576888e-34 - 5.55699345e-34j, 1.78838516e-35 + 2.57051097e-35j,
            #                       7.94312070e-21 + 6.23110325e-21j, 1.69415215e-18 + 1.32900372e-18j,
            #                       -8.07644298e-36 - 1.54484421e-35j, -1.76920367e-34 - 1.34121671e-34j,
            #                       1.84998335e-35 + 0.00000000e+00j, -2.86639386e-35 - 1.75030193e-34j,
            #                       -3.73349102e-21 - 2.92879447e-21j, 1.68994252e-18 + 1.32570141e-18j,
            #                       -6.04626595e-35 - 1.29190865e-34j, -1.99366459e-35 + 6.25317067e-36j,
            #                       1.68994252e-18 + 1.32570141e-18j, -3.73349102e-21 - 2.92879447e-21j],
            #                      [-1.51988854e-17 + 1.39694931e-17j, -7.12607671e-20 + 6.54966968e-20j,
            #                       6.40348195e-33 - 6.13625811e-33j, -2.70910110e-34 + 1.29374370e-34j,
            #                       -7.12607671e-20 + 6.54966968e-20j, -1.51988854e-17 + 1.39694931e-17j,
            #                       1.58674209e-34 - 5.24765899e-35j, 1.54307208e-33 - 1.46606458e-33j,
            #                       -2.86639386e-35 + 1.75030193e-34j, 1.70040394e-33 + 0.00000000e+00j,
            #                       3.34945727e-20 - 3.07852969e-20j, -1.51611192e-17 + 1.39347817e-17j,
            #                       1.31597940e-33 - 3.71877505e-34j, -2.82721936e-35 - 1.98312891e-34j,
            #                       -1.51611192e-17 + 1.39347817e-17j, 3.34945727e-20 - 3.07852969e-20j],
            #                      [-5.52301217e-04 + 1.50603299e-17j, -2.58949306e-06 + 4.56681157e-20j,
            #                       2.37230937e-19 - 4.93897365e-21j, -7.67867356e-21 - 2.35632772e-21j,
            #                       -2.58949306e-06 + 6.70640448e-20j, -5.52301217e-04 + 1.50277377e-17j,
            #                       4.07563879e-21 + 1.83906402e-21j, 5.69381023e-20 - 9.41702512e-22j,
            #                       -3.73349102e-21 + 2.92879447e-21j, 3.34945727e-20 + 3.07852969e-20j,
            #                       1.21713486e-06 + 0.00000000e+00j, -5.50928859e-04 + 1.49294996e-17j,
            #                       3.26549040e-20 + 1.65001609e-20j, 3.03348873e-21 - 4.41822867e-21j,
            #                       -5.50928859e-04 + 1.49672074e-17j, 1.21713486e-06 - 5.62836326e-20j],
            #                      [2.49995863e-01 - 4.23860698e-17j, 1.17211864e-03 + 1.10915833e-17j,
            #                       -1.07381174e-16 + 2.23559707e-18j, 3.47570594e-18 + 1.06657773e-18j,
            #                       1.17211864e-03 + 1.40684324e-18j, 2.49995863e-01 - 2.76333864e-17j,
            #                       -1.84481367e-18 - 8.32441395e-19j, -2.57726935e-17 + 4.26256044e-19j,
            #                       1.68994252e-18 - 1.32570141e-18j, -1.51611192e-17 - 1.39347817e-17j,
            #                       -5.50928859e-04 - 1.49294996e-17j, 2.49374673e-01 + 0.00000000e+00j,
            #                       -1.47810482e-17 - 7.46869975e-18j, -1.37309064e-18 + 1.99988494e-18j,
            #                       2.49374673e-01 - 1.70681926e-17j, -5.50928859e-04 + 1.05469521e-17j],
            #                      [-1.48178676e-17 + 7.48730420e-18j, -6.94743451e-20 + 3.51046162e-20j,
            #                       6.29778989e-33 - 3.34854458e-33j, -2.37957307e-34 + 4.08777166e-35j,
            #                       -6.94743451e-20 + 3.51046162e-20j, -1.48178676e-17 + 7.48730420e-18j,
            #                       1.34278009e-34 - 5.91079689e-36j, 1.51484448e-33 - 7.97150000e-34j,
            #                       -6.04626595e-35 + 1.29190865e-34j, 1.31597940e-33 + 3.71877505e-34j,
            #                       3.26549040e-20 - 1.65001609e-20j, -1.47810482e-17 + 7.46869975e-18j,
            #                       1.09979437e-33 + 0.00000000e+00j, 2.14904694e-35 - 1.59661753e-34j,
            #                       -1.47810482e-17 + 7.46869975e-18j, 3.26549040e-20 - 1.65001609e-20j],
            #                      [-1.37651100e-18 - 2.00486664e-18j, -6.45384361e-21 - 9.39992179e-21j,
            #                       6.09183842e-34 + 8.48844479e-34j, -1.05841805e-35 - 3.37464899e-35j,
            #                       -6.45384361e-21 - 9.39992179e-21j, -1.37651100e-18 - 2.00486664e-18j,
            #                       3.48194694e-36 + 1.93782011e-35j, 1.45326335e-34 + 2.04339650e-34j,
            #                       -1.99366459e-35 - 6.25317067e-36j, -2.82721936e-35 + 1.98312891e-34j,
            #                       3.03348873e-21 + 4.41822867e-21j, -1.37309064e-18 - 1.99988494e-18j,
            #                       2.14904694e-35 + 1.59661753e-34j, 2.35986984e-35 + 0.00000000e+00j,
            #                       -1.37309064e-18 - 1.99988494e-18j, 3.03348873e-21 + 4.41822867e-21j],
            #                      [2.49995863e-01 - 2.52753605e-17j, 1.17211864e-03 + 1.11718078e-17j,
            #                       -1.07381174e-16 + 2.23559707e-18j, 3.47570594e-18 + 1.06657773e-18j,
            #                       1.17211864e-03 + 1.48706770e-18j, 2.49995863e-01 - 1.05226771e-17j,
            #                       -1.84481367e-18 - 8.32441395e-19j, -2.57726935e-17 + 4.26256044e-19j,
            #                       1.68994252e-18 - 1.32570141e-18j, -1.51611192e-17 - 1.39347817e-17j,
            #                       -5.50928859e-04 - 1.49672074e-17j, 2.49374673e-01 + 1.70681926e-17j,
            #                       -1.47810482e-17 - 7.46869975e-18j, -1.37309064e-18 + 1.99988494e-18j,
            #                       2.49374673e-01 + 0.00000000e+00j, -5.50928859e-04 + 1.05092444e-17j],
            #                      [-5.52301217e-04 - 1.04795834e-17j, -2.58949306e-06 - 7.40770999e-20j,
            #                       2.37230937e-19 - 4.93897365e-21j, -7.67867356e-21 - 2.35632772e-21j,
            #                       -2.58949306e-06 - 5.26811708e-20j, -5.52301217e-04 - 1.05121757e-17j,
            #                       4.07563879e-21 + 1.83906402e-21j, 5.69381023e-20 - 9.41702512e-22j,
            #                       -3.73349102e-21 + 2.92879447e-21j, 3.34945727e-20 + 3.07852969e-20j,
            #                       1.21713486e-06 + 5.62836326e-20j, -5.50928859e-04 - 1.05469521e-17j,
            #                       3.26549040e-20 + 1.65001609e-20j, 3.03348873e-21 - 4.41822867e-21j,
            #                       -5.50928859e-04 - 1.05092444e-17j, 1.21713486e-06 + 0.00000000e+00j]])

            log_f_Sproj_i = np.float64(116.06925990904078)
            # todo: S_proj_i is slgihtly different and causes fidelity to fall from 0.9999 -> 0.978. Going to next step to solve

            delta_i = S_proj_i - S

            frob_gradf_delta_i = TomographyMLE._frobenius_inner_product(grad_log_f_S, delta_i)
            some_param = (1 / (2 * t_i)) * np.linalg.norm(delta_i, ord='fro') ** 2

            print(' lhs', log_f_Sproj_i)
            print(' rhs', log_f_S + frob_gradf_delta_i + some_param)

            while log_f_Sproj_i > (log_f_S + frob_gradf_delta_i + some_param):
                print('stuck in while loop')

                t_i *= beta
                S_proj_i = TomographyMLE._proj(S - t_i * grad_log_f_S)
                delta_i = S_proj_i - S

            delta_i_hat = S_proj_i - S_proj_i1

            if TomographyMLE._frobenius_inner_product(delta_i, delta_i_hat) < 0:
                S_proj_i, S, theta = copy.deepcopy(S_proj_i1), copy.deepcopy(S_proj_i1), 1
            else:
                theta, S = (1 + np.sqrt(1 + 4 * theta ** 2)) / 2, S_proj_i + delta_i_hat * (theta - 1) / (
                            (1 + np.sqrt(1 + 4 * theta ** 2)) / 2)

            print('What is this F',
                  np.abs(self._log_likelihood_func(S_proj_i) - self._log_likelihood_func(S_proj_i1))                  )

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
        choi = self._choi_matrix(S_0, beta, t, max_it)
        X = np.zeros((len(choi), len(choi)), dtype='complex_')

        for m in range(len(choi)):
            P_m = np.conjugate(np.transpose(_matrix_to_vector(np.transpose(_get_fixed_basis_ops(m, self._nqubit)))))

            for n in range(len(choi)):
                X[m, n] = (1 / 2 ** self._nqubit) * np.linalg.multi_dot(
                    [P_m, choi, _matrix_to_vector(np.transpose(_get_fixed_basis_ops(n, self._nqubit)))])
        return X
