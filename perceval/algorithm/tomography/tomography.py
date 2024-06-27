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
from collections import defaultdict

from perceval.components import AProcessor, PauliType
from perceval.utils import BasicState

from .abstract_process_tomography import AProcessTomography
from .tomography_utils import (_matrix_basis, _matrix_to_vector, _vector_to_sq_matrix, _coef_linear_decomp,
                               _get_fixed_basis_ops, _get_canonical_basis_ops, _generate_pauli_index,
                               _generate_pauli_prep_index, _list_subset_k_from_n, _compute_probs)
from ..abstract_algorithm import AAlgorithm


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

        self._size_hilbert = 2 ** self._nqubit
        self._gate_logical_perf = None
        self._qst_cache = defaultdict(lambda: defaultdict(lambda: dict))

    _LOGICAL0 = BasicState([1, 0])
    _LOGICAL1 = BasicState([0, 1])

    def _stokes_parameter(self, prep_state_indices: list, meas_pauli_basis_indices: list) -> float:
        """
        Computes the Stokes parameter S_i for state prep_state_indices after operator_circuit
        :param prep_state_indices: list of length of number of qubits representing the preparation circuit
        :param meas_pauli_basis_indices: list of length of number of qubits representing the measurement circuit and
        the eigenvectors being measured
        :return: Value of Stokes parameter for a given combination of input and output state -> a complex float
        """

        if PauliType.Z not in meas_pauli_basis_indices:
            output_distribution, self._gate_logical_perf = _compute_probs(self, prep_state_indices,
                                                                          meas_pauli_basis_indices)
            self._qst_cache[tuple(prep_state_indices)][tuple(meas_pauli_basis_indices)] = output_distribution
        else:
            meas_indices_Z_to_I = [elem if elem != PauliType.Z else PauliType.I for elem in meas_pauli_basis_indices]
            output_distribution = self._qst_cache[tuple(prep_state_indices)][tuple(meas_indices_Z_to_I)]

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

        pauli_meas_indices = _generate_pauli_index(self._nqubit)  # generates indices for measurement

        for index, elem in enumerate(pauli_meas_indices):
            density_matrix += self._stokes_parameter(prep_state_indices, elem) \
                              * _get_fixed_basis_ops(index, self._nqubit)
        density_matrix = ((1 / 2) ** self._nqubit) * density_matrix

        return density_matrix


class ProcessTomography(AProcessTomography):
    """
    Experiment to reconstruct the process map of the gate operation by tomography experiment.
    - Computes the mathematical tensors/matrices defined by theory required to perform process tomography

    - Computes Chi matrix form of the operation process map

    - Provides analysis methods to investigate the results of process tomography

        -- Fidelity of the operation, Error process map

    """
    def __init__(self, operator_processor: AProcessor, **kwargs):
        """

        :param operator_processor: A perceval Processor with gate (or operation) on which process tomography
        needs to be performed. By default, it will have a perfect source and use the SLOSBackend() for computations.
        """
        super().__init__(processor=operator_processor, **kwargs)
        self._qst = StateTomography(operator_processor=self._processor, **kwargs)

        self.chi_normalized = None
        self.chi_unnormalized = None
        self.gate_efficiency = None
        self._prep_basis_size = 4
        # Standard Process tomography works with a subset of pauli eigenstates prepared at input : |0>, |1>, |+>, |i+>

    def _lambda_vector(self) -> np.ndarray:
        """
        Computes the lambda vector of the operator
        :return: Lambda vector for Chi computation
        """
        density_matrices = []  # stores a list of density matrices for each measurement

        pauli_prep_indices = _generate_pauli_prep_index(self._nqubit, self._prep_basis_size)
        for prep_state_indices in pauli_prep_indices:
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

        pauli_prep_indices = _generate_pauli_prep_index(self._nqubit, self._prep_basis_size)
        for prep_state_indices in pauli_prep_indices:
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
