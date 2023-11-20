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

import numpy as np

from perceval.components import AProcessor, Processor, PauliType
from perceval.utils import BasicState

from .tomography_utils import _matrix_basis, _matrix_to_vector, _vector_to_sq_matrix, _coef_linear_decomp, \
    _get_fixed_basis_ops, _get_canonical_basis_ops, _krauss_repr_ops, _generate_pauli_index, _list_subset_k_from_n
from ._prep_n_meas_circuits import StatePreparation, MeasurementCircuit
from ..abstract_algorithm import AAlgorithm
from ..sampler import Sampler


class StateTomography(AAlgorithm):
    """
    Experiment to reconstruct the state of the system by tomography experiment.

    - Adds preparation and measurement circuits to input processor (with the gate operation under study)

    - Computes parameters required to do state tomography

    - Performs Tomography experiment - Computes and Returns density matrices for each input state
    """
    def __init__(self, operator_processor: AProcessor, **kwargs):
        """
        :param operator_processor: A perceval Processor with gate (or operation) on which state tomography
        needs to be performed. By default, it will have a perfect source and use the SLOSBackend() for computations.
        """
        super().__init__(processor=operator_processor, **kwargs)
        self._nqubit, odd_modes = divmod(operator_processor.m, 2)
        if odd_modes:
            raise ValueError(
                f"Input processor has an odd mode count ({operator_processor.m}) and thus, is not a logical gate")

        if self._processor.is_remote:
            raise TypeError("Tomography does not support Remote Processor yet")

        self._size_hilbert = 2 ** self._nqubit
        self._gate_logical_perf = None

    _LOGICAL0 = BasicState([1, 0])
    _LOGICAL1 = BasicState([0, 1])

    def _configure_processor(self, prep_state_indices: list, meas_pauli_basis_indices: list) -> Processor:
        """
        Adds preparation and measurement circuit to input processor (with the gate operation under study) and
        computes the output probability distribution
        :param prep_state_indices: List of "nqubit" indices selecting the circuit at each qubit for a preparation state
        :param meas_pauli_basis_indices: List of "nqubit" indices selecting the circuit at each qubit for a measurement
         circuit
        :return: the configured processor to perform state tomography experiment
        """

        p = self._processor.copy()
        p.clear_input_and_circuit(self._nqubit*2)  # Clear processor content but keep its size

        pc = StatePreparation(prep_state_indices)
        for c in pc:
            p.add(*c)  # Add state preparation circuit to the left of the operator

        p.add(0, self._processor)  # including the operator (as a processor)

        mc = MeasurementCircuit(meas_pauli_basis_indices)
        for c in mc:
            p.add(*c)  # Add measurement basis circuit to the right of the operator

        p.min_detected_photons_filter(0)  # QPU would have a problem with this - Eric

        input_state = BasicState([1, 0]*self._nqubit)
        p.with_input(input_state)

        return p

    def _compute_probs(self, prep_state_indices: list, meas_pauli_basis_indices: list) -> dict:
        """
        computes the output probability distribution for the state tomography experiment
        :param prep_state_indices: List of "nqubit" indices selecting the circuit at each qubit for a preparation state
        :param meas_pauli_basis_indices: List of "nqubit" indices selecting the circuit at each qubit for a measurement
         circuit
        :return: Output state probability distribution
        """

        p = self._configure_processor(prep_state_indices, meas_pauli_basis_indices)
        sampler = Sampler(p, max_shots_per_call=self._max_shots)
        probs = sampler.probs()
        output_distribution = probs["results"]
        self._gate_logical_perf = probs["logical_perf"]

        for key in output_distribution:  # Denormalize output state distribution
            output_distribution[key] *= self._gate_logical_perf
        return output_distribution

    def _stokes_parameter(self, prep_state_indices: list, meas_pauli_basis_indices: list) -> float:
        """
        Computes the Stokes parameter S_i for state prep_state_indices after operator_circuit
        :param prep_state_indices: list of length of number of qubits representing the preparation circuit
        :param meas_pauli_basis_indices: list of length of number of qubits representing the measurement circuit and
        the eigenvectors being measured
        :return: Value of Stokes parameter for a given combination of input and output state -> a complex float
        """

        output_distribution = self._compute_probs(prep_state_indices, meas_pauli_basis_indices)

        # calculation of the Stokes parameter begins here
        stokes_param = 0
        for k in range(self._nqubit + 1):
            for J in _list_subset_k_from_n(k, self._nqubit):
                eta = 1
                measurement_state = BasicState()
                for j in range(0, self._nqubit):
                    if j not in J:
                        measurement_state *= self._LOGICAL0
                    else:
                        measurement_state *= self._LOGICAL1
                        if meas_pauli_basis_indices[j] != PauliType.I:
                            eta *= -1
                stokes_param += eta * output_distribution[measurement_state]

        return stokes_param

    def perform_state_tomography(self, prep_state_indices: list) -> np.ndarray:
        """
        Computes the density matrix of a state after the operator_circuit. Size d x d where d=size_of_hilbert_space

        :param prep_state_indices: list of length of number of qubits to index the corresponding preparation circuit
        :return: density matrix for a given input state preparation. size_hilbert x size_hilbert array.
        """
        density_matrix = np.zeros((self._size_hilbert, self._size_hilbert), dtype='complex_')

        pauli_indices = _generate_pauli_index(self._nqubit)
        for index, elem in enumerate(pauli_indices):
            density_matrix += self._stokes_parameter(prep_state_indices, elem) \
                              * _get_fixed_basis_ops(index, self._nqubit)
        density_matrix = ((1 / 2) ** self._nqubit) * density_matrix

        return density_matrix


class ProcessTomography(AAlgorithm):
    """
    Experiment to reconstruct the process map of the gate operation by tomography experiment.
    - Computes the mathematical tensors/matrices defined by theory required to perform process tomography

    - Computes :math:'$\chi$' matrix form of the operation process map

    - Provides analysis methods to investigate the results of process tomography

        -- Fidelity of the operation, Error process map

    """
    def __init__(self, operator_processor: AProcessor, **kwargs):
        """

        :param operator_processor: A perceval Processor with gate (or operation) on which process tomography
        needs to be performed. By default, it will have a perfect source and use the SLOSBackend() for computations.
        """
        super().__init__(processor=operator_processor, **kwargs)
        self._nqubit = operator_processor.m // 2
        if self._nqubit > 3:
            raise ValueError(
                f"Input gate too large. Tomography supports up to 3-qubit gates ({self._nqubit}-qubit gate passed).")

        self._size_hilbert = 2 ** self._nqubit
        self._qst = StateTomography(operator_processor=self._processor, **kwargs)

        self.chi_normalized = None
        self.chi_unnormalized = None
        self.gate_efficiency = None

    def _beta_tensor_elem(self, j: int, k: int, m: int, n: int, nqubit: int) -> np.ndarray:
        """
        computes the elements of beta^{mn}_{jk}, a rank 4 tensor, each index of which can
        take values between 0 and d^2-1  [d = _size_hilbert]

        :param j: one of the indices for the beta tensor, value between 0 and d**2-1
        :param k: one of the indices for the beta tensor, value between 0 and d**2-1
        :param m: one of the indices for the beta tensor, value between 0 and d**2-1
        :param n: one of the indices for the beta tensor, value between 0 and d**2-1
        :param nqubit: numbero f qubits
        :return:
        """

        b = _krauss_repr_ops(m, _get_canonical_basis_ops(j, self._size_hilbert), n, nqubit)
        q, r = divmod(k, self._size_hilbert)  # quotient, remainder
        return b[q, r]

    def _beta_as_matrix(self) -> np.ndarray:
        """
        compiles the 2D beta matrix by extracting elements of the rank 4 tensor computed by method _beta_tensor_elem
        :return: Beta Matrix for Chi computation
        """
        num_meas = self._size_hilbert ** 4  # Total number of measurements needed for process tomography
        beta_matrix = np.zeros((num_meas, num_meas), dtype='complex_')
        for a in range(num_meas):
            j, k = divmod(a, self._size_hilbert ** 2)  # returns quotient, remainder
            for b in range(num_meas):
                # j,k,m,n are indices for _beta_tensor_elem
                m, n = divmod(b, self._size_hilbert ** 2)
                beta_matrix[a, b] = self._beta_tensor_elem(j, k, m, n, self._nqubit)
        return beta_matrix

    def _lambda_vector(self) -> np.ndarray:
        """
        Computes the lambda vector of the operator
        :return: Lambda vector for Chi computation
        """
        density_matrices = []  # stores a list of density matrices for each measurement
        pauli_indices = _generate_pauli_index(self._nqubit)

        for prep_state_indices in pauli_indices:
            # compute state of system for each preparation state - perform state tomography
            density_matrices.append(self._qst.perform_state_tomography(prep_state_indices))

        # this creates the fixed basis for the Pauli states prepared 0, 1, + and i

        lambda_matrix = np.zeros((self._size_hilbert ** 2, self._size_hilbert ** 2), dtype='complex_')

        for j in range(self._size_hilbert ** 2):
            rhoj = _get_canonical_basis_ops(j, self._size_hilbert)
            mu = _coef_linear_decomp(rhoj, _matrix_basis(self._nqubit, self._size_hilbert))
            eps_rhoj = sum([mu[i] * density_matrices[i] for i in range(self._size_hilbert ** 2)])
            for k in range(self._size_hilbert ** 2):
                quotient, remainder = divmod(k, self._size_hilbert)
                lambda_matrix[j, k] = eps_rhoj[quotient, remainder]
        return _matrix_to_vector(lambda_matrix)

    def _lambda_target(self, operator: np.ndarray) -> np.ndarray:
        """
        Implements a mathematical formula for ideal gate (given operator) to compute process fidelity
        :param operator: Target operator matrix
        :return: lambda vector to compute chi for the target operator
        """
        lambda_matrix = np.zeros((self._size_hilbert ** 2, self._size_hilbert ** 2), dtype='complex_')
        for j in range(self._size_hilbert ** 2):
            rhoj = _get_canonical_basis_ops(j, self._size_hilbert)
            eps_rhoj = np.linalg.multi_dot([operator, rhoj, np.conjugate(np.transpose(operator))])
            for k in range(self._size_hilbert ** 2):
                quotient, remainder = divmod(k, self._size_hilbert)
                lambda_matrix[j, k] = eps_rhoj[quotient, remainder]

        L1 = np.zeros((self._size_hilbert ** 4, 1), dtype='complex_')
        for i in range(self._size_hilbert ** 4):
            quotient, remainder = divmod(i, self._size_hilbert ** 2)
            L1[i] = lambda_matrix[quotient, remainder]
        return L1

    def chi_matrix(self) -> np.ndarray:
        """
        Computes the chi matrix of the operator_circuit. Size d^4 x d^4 [=2**(2*nqubit)x2**(2*nqubit) array]
        :return: Chi matrix normalized by gate efficiency (=its trace)
        """
        if self.chi_normalized is None:
            beta_inv = np.linalg.pinv(self._beta_as_matrix())
            L = self._lambda_vector()
            X = np.dot(beta_inv, L)  # X is a vector here
            self.chi_unnormalized = _vector_to_sq_matrix(X)
            self.gate_efficiency = np.trace(self.chi_unnormalized)
            self.chi_normalized = self.chi_unnormalized / self.gate_efficiency
        return self.chi_normalized  # always returns normalized chi map

    def chi_target(self, operator: np.ndarray) -> np.ndarray:
        """
        Implements a mathematical formula for ideal gate (given operator) to compute process fidelity
        :param operator: Target operator matrix
        :return: Target Chi matrix
        """

        beta_inv = np.linalg.pinv(self._beta_as_matrix())
        lambd = self._lambda_target(operator)
        X = np.dot(beta_inv, lambd)  # X is a matrix here
        chi = _vector_to_sq_matrix(X[:, 0])
        return chi

    @staticmethod
    def process_fidelity(chi_computed: np.ndarray, chi_ideal: np.ndarray) -> float:
        """
        Computes the process fidelity of an operator (ideal) and its implementation (realistic)

        :param chi_computed: chi matrix computed from process tomography
        :param chi_ideal: Ideal chi matrix for the corresponding operator
        :return: float between 0 and 1
        """
        return np.real(np.trace(np.dot(chi_computed, chi_ideal)))

    def average_fidelity(self, operator: np.ndarray) -> float:
        """
        Computes the average fidelity of an operator (ideal) and its implementation (realistic).
        This is not a full fidelity of the operation as given by the process_fidelity but
        simply that of the gate.

        :param operator: operator (gate) matrix whose fidelity is to be calculated
        :return: float between 0 and 1
        """
        Udag = np.transpose(np.conjugate(operator))
        avg_fidelity = 1 / (self._size_hilbert + 1)

        # compute the map on a basis of states (tensor products of |0>, |1>, |+>,|i+>)
        density_matrices = []   # stores a list of density matrices for each measurement
        pauli_indices = _generate_pauli_index(self._nqubit)
        for prep_state_indices in pauli_indices:
            density_matrices.append(self._qst.perform_state_tomography(prep_state_indices))
            # setting values

        density_matrices = [x / self._qst._gate_logical_perf for x in density_matrices]

        for j in range(self._size_hilbert ** 2):
            Uj = _get_fixed_basis_ops(j, self._nqubit)
            mu = _coef_linear_decomp(Uj, _matrix_basis(self._nqubit, self._size_hilbert))
            eps_Uj = sum([mu[i] * density_matrices[i] for i in range(self._size_hilbert ** 2)])
            # compute the map on a basis
            Ujdag = np.transpose(np.conjugate(Uj))
            a = np.linalg.multi_dot([operator, Ujdag, Udag, eps_Uj])
            avg_fidelity += (1 / ((self._size_hilbert + 1) * (self._size_hilbert ** 2))) * np.trace(a)
        return np.real(avg_fidelity)

    def error_process_matrix(self, computed_chi: np.ndarray, operator: np.ndarray) -> np.ndarray:
        """
        Computes the error matrix for an operation from the computed chi
        Size d^4 x d^4

        :param computed_chi: chi matrix computed from process tomography
        :param operator: Gate (or operator) matrix
        :return: error process matrix
        """
        V = np.zeros((self._size_hilbert ** 2, self._size_hilbert ** 2), dtype='complex_')
        for m in range(self._size_hilbert ** 2):
            for n in range(self._size_hilbert ** 2):
                Emdag = np.transpose(np.conjugate(_get_fixed_basis_ops(m, self._nqubit)))
                En = _get_fixed_basis_ops(n, self._nqubit)
                Udag = np.transpose(np.conjugate(operator))
                V[m, n] = (1 / self._size_hilbert) * np.trace(np.linalg.multi_dot([Emdag, En, Udag]))
        return np.linalg.multi_dot([V, computed_chi, np.conjugate(np.transpose(V))])
