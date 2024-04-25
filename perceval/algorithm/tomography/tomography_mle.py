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
from scipy.linalg import sqrtm
from .tomography_utils import _state_to_dens_matrix, _matrix_to_vector, _get_fixed_basis_ops, _compute_probs, \
    _generate_pauli_prep_index, _generate_pauli_index
from perceval.utils import BasicState
from perceval.components import AProcessor, get_pauli_gate, PauliType, PauliEigenStateType
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

    @abstractmethod
    def _povm_data(self):
        pass

    def _collect_data(self, prep_state_indices, state_meas_indices):
        # performs measurements on the output_state for given preparation and measurement state indices at each qubit

        output_distribution, self._gate_logical_perf = _compute_probs(self, prep_state_indices, state_meas_indices,
                                                                      denormalize=False)

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

    def _povm_state(self):
        # Gives a set of measurement states, so they form a set of informationally complete measurements.
        # For 1 qubit, they are (order important) : |0>,|1>,|+>,|->,|i+>,|i->,
        # These measurement states are eigenvectors of the tensor products of Pauli operators

        d = 2 ** self._nqubit
        P = []
        B = [np.array([[1], [0]], dtype='complex_'), np.array([[0], [1]], dtype='complex_'),
             (1 / np.sqrt(2)) * np.array([[1], [1]], dtype='complex_'),
             (1 / np.sqrt(2)) * np.array([[1], [-1]], dtype='complex_'),
             (1 / np.sqrt(2)) * np.array([[1], [1j]], dtype='complex_'),
             (1 / np.sqrt(2)) * np.array([[1], [-1j]], dtype='complex_')]  #

        for pauli_index in range(3 ** self._nqubit):
            # iterates over pauli operators I(equivalent to Z),X,Y or their tensor products
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
        # Gives a POVM (positive operator value measure) set suited for tomography
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

    def _input_basis(self):
        # Computes input density matrix basis (similar to POVM but not same order)

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
                M = np.kron(self._compute_matrix(i // 2), M)

                if i % 2 == 0:
                    v = np.kron(np.array([[1], [0]], dtype='complex_'), v)
                else:
                    v = np.kron(np.array([[0], [1]], dtype='complex_'), v)
            B.append(_state_to_dens_matrix(np.dot(M, v)))
        return B

    @staticmethod
    def _frobenius_inner_product(A, B):
        # calculates the inner product associated to Frobenius norm
        return np.trace(np.dot(np.transpose(np.conjugate(A)), B))

    @staticmethod
    def _proj_simplex(eigenvalues):
        # Projects a given real eigen-spectra (sorted in descending order) on positive elements with their sum equal to 1
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
        # Projects a given hermitian matrix on the cone of positive semi-definite trace=1 matrices
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

    def _perform_mle_tomography(self, init_guess_quantum_map, decelerate_factor: float=0.5, init_learn_rate:float=1,
                                max_iterations :int=1000, convergence_precision:float=1e-10):
        # Accelerated Projected Gradient descent algorithm which takes an input guess and
        # uses measurements to reconstruct quantum maps (state or process) using MLE for Quantum Tomography
        # ref: https://doi.org/10.48550/arXiv.1609.07881
        #
        # param init_guess_quantum_map: an initial guess ofr the quantum map
        # param decelerate_factor: Decreases the learning rate for a slower descent with a value in range (0,1)
        # param init_learn_rate: initial learning rate
        # :param max_iterations: maximum number of iterations

        guess_quantum_map = copy.deepcopy(init_guess_quantum_map)
        init_quantum_map = copy.deepcopy(init_guess_quantum_map)

        theta = 1
        ith_learn_rate = init_learn_rate  # initializing the learning rate of the algorithm

        log_f_guess_quantum_map = self._log_likelihood_func(guess_quantum_map)
        grad_log_f_guess_quantum_map = self._grad_log_likelihood_func(guess_quantum_map)

        for i in range(max_iterations):
            ith_quantum_map = self._proj(guess_quantum_map - ith_learn_rate * grad_log_f_guess_quantum_map)

            delta_i = ith_quantum_map - guess_quantum_map  # difference between current and target

            log_f_ith_quantum_map = self._log_likelihood_func(ith_quantum_map)

            frob_prod_grad_log_f_delta = self._frobenius_inner_product(grad_log_f_guess_quantum_map, delta_i)

            norm_delta_i = (1 / (2 * ith_learn_rate)) * np.linalg.norm(delta_i, ord='fro') ** 2


            while log_f_ith_quantum_map > (log_f_guess_quantum_map + frob_prod_grad_log_f_delta + norm_delta_i):
                ith_learn_rate *= decelerate_factor
                ith_quantum_map = TomographyMLE._proj(guess_quantum_map - ith_learn_rate * grad_log_f_guess_quantum_map)
                delta_i = ith_quantum_map - guess_quantum_map

            delta_i_hat = ith_quantum_map - init_quantum_map

            if TomographyMLE._frobenius_inner_product(delta_i, delta_i_hat) < 0:
                # Restart by re-initializing guess maps
                ith_quantum_map, guess_quantum_map, theta = init_quantum_map, init_quantum_map, 1
            else:
                # Accelerate the algorithm in the next iteration
                theta, guess_quantum_map = (1 + np.sqrt(1 + 4 * theta ** 2)) / 2, ith_quantum_map + delta_i_hat * (theta - 1) / (
                            (1 + np.sqrt(1 + 4 * theta ** 2)) / 2)

            if np.abs(log_f_ith_quantum_map - self._log_likelihood_func(init_quantum_map)) < convergence_precision:
                break

            init_quantum_map = ith_quantum_map

        return ith_quantum_map



class StateTomographyMLE(TomographyMLE):
    def __init__(self, operator_processor):
        super().__init__(operator_processor)
        self._f = self._povm_data()
        self._guess_density_matrix = np.eye(2 ** self._nqubit) / (2 ** self._nqubit)

    def _povm_data(self):
        # Performing a POVM (positive operator value measure) on the quantum processor
        # in the informationally complete Pauli basis, i.e. choosing all the eigenvectors of the PAuli operators.
        # They are |0>,|1>,|+>,|->,|i+>,|i->

        f = []
        measurement_indices = _generate_pauli_index(self._nqubit)

        preparation_indices = [PauliEigenStateType.Zm]  * self._nqubit
        # Input Preparation fixed to |0> for state tomography

        for val in measurement_indices:
            if PauliType.Z in val:
                continue
            f += self._collect_data(preparation_indices, state_meas_indices=val)

        return f

    def _log_likelihood_func(self, rho: np.ndarray) -> float:
        # Log-likelihood function of the POVM to minimize

        f = self._f
        P = self._povm_operator()

        x = 0
        for k in range(len(f)):
            if np.trace(np.dot(rho, P[k])) != 0:
                x -= f[k] * np.log(np.trace(np.dot(rho, P[k])))
        return x

    def _grad_log_likelihood_func(self, rho: np.ndarray) -> float:
        # Gradient of the log-likelihood function of the POVM

        f = self._f
        P = self._povm_operator()

        grad = 0
        for k in range(len(f)):
            if np.trace(np.dot(rho, P[k])) != 0:
                grad -= (f[k] / (np.trace(np.dot(rho, P[k])))) * P[k]
        return grad

    def state_tomography_density_matrix(self, decelerate_factor: float = 0.5, init_learn_rate: float = 1,
                                        max_iterations: int = 1000):

        return self._perform_mle_tomography(self._guess_density_matrix, 0.5, 1, 1000)


    @staticmethod
    def state_fidelity(x, y):
        """
        Computes the fidelity of the density matrix reconstructed after State Tomography using MLE algorithm
        """
        rx = sqrtm(x)
        z = np.linalg.multi_dot([rx, y, rx])
        return np.real(np.trace(sqrtm(z)) ** 2)



class ProcessTomographyMLE(TomographyMLE):
    def __init__(self, operator_processor):
        super().__init__(operator_processor)
        self._f = self._povm_data()
        self._guess_choi_seed = np.eye((2 ** self._nqubit), dtype='complex_')
        self._guess_choi_matrix = np.kron(self._guess_choi_seed, self._guess_choi_seed) / 16

    def _povm_data(self):
        # Performing a POVM (positive operator value measure) on the quantum processor
        # in the informationally complete Pauli basis, i.e. choosing all the eigenvectors of the Pauli operators.
        # They are |0>,|1>,|+>,|->,|i+>,|i->

        # measurement is always on 3 Paulitype I, X, Y : Z is moved away
        # prep has 6 options -> 4 pauli and 2 other are some combo
        # of that itself -> find which and decide how to implement

        f = []
        preparation_states = _generate_pauli_prep_index(self._nqubit)
        measurement_states = _generate_pauli_index(self._nqubit)

        for index, value in enumerate(preparation_states):
            f_per_prep = []
            for meas_indices in measurement_states:
                if PauliType.Z in meas_indices:
                    continue
                f_per_prep += self._collect_data(prep_state_indices=value, state_meas_indices=meas_indices)

            f.append(f_per_prep)
        return f

    def _log_likelihood_func(self, choi_matrix: np.ndarray) -> float:
        # Log-likelihood function of the POVM to minimize

        f = self._f
        P = self._povm_operator()
        B = self._input_basis()

        x = 0
        for m in range(len(B)):
            for l in range(len(P)):
                pml = 2 ** self._nqubit * np.real(np.trace(np.dot(choi_matrix, np.kron(np.transpose(B[m]), P[l]))))
                if 0 < pml <= 1:
                    x -= f[m][l] * np.log(pml)
        return x

    def _grad_log_likelihood_func(self, S) -> float:
        # Gradient of the log-likelihood function of the POVM

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

    def chi_matrix(self, decelerate_factor: float=0.5, init_learn_rate:float=1, max_iterations :int=100):
        """
        Computes the chi matrix of the quantum process under study using the MLE tomography
        # param decelerate_factor: Decreases the learning rate for a slower descent with a value in range (0,1)
        # param init_learn_rate: initial learning rate
        # :param max_iterations: maximum number of iterations

        :return: chi matrix
        """
        choi = self._perform_mle_tomography(self._guess_choi_matrix, decelerate_factor, init_learn_rate, max_iterations)
        X = np.zeros((len(choi), len(choi)), dtype='complex_')

        for m in range(len(choi)):
            P_m = np.conjugate(np.transpose(_matrix_to_vector(np.transpose(_get_fixed_basis_ops(m, self._nqubit)))))

            for n in range(len(choi)):
                X[m, n] = (1 / 2 ** self._nqubit) * np.linalg.multi_dot(
                    [P_m, choi, _matrix_to_vector(np.transpose(_get_fixed_basis_ops(n, self._nqubit)))])
        return X
