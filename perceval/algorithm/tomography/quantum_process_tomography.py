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
from itertools import combinations

from perceval.components import Circuit, Processor, BS, PS, PERM, Port
from perceval.algorithm.tomography.abstract_tomography import ATomography
from perceval.utils import BasicState, Encoding
from perceval.utils.postselect import PostSelect
from typing import List
from perceval.algorithm.tomography._tomography_utils import state_to_dens_matrix, compute_matrix, matrix_basis, \
    matrix_to_vector, vector_to_matrix, decomp


def pauli(j):
    """
    computes the j-th Pauli operator (I,X,Y,Z)

    :param j: int between 0 and 3
    :return: 2x2 unitary and hermitian array
    """
    # assert j < 3, f'Pauli Index should be less than 3'
    if j == 0:  # I
        return np.eye(2, dtype='complex_')
    elif j == 1:  # Pauli X
        return np.array([[0, 1], [1, 0]], dtype='complex_')
    elif j == 2:  # Pauli Y
        return np.array([[0, -1j], [1j, 0]], dtype='complex_')
    else:  # Pauli Z
        return np.array([[1, 0], [0, -1]], dtype='complex_')


def fixed_basis_ops(j, nqubit):
    """
    computes the fixed sets of operators (tensor products of pauli gates) for quantum process tomography

    :param j: int between 0 and 4**nqubit-1
    :param nqubit: number of qubits
    :return: 2**nqubit x 2**nqubit array
    """
    if nqubit == 1:
        return pauli(j)

    E = pauli(j // (4 ** (nqubit - 1)))
    j = j % (4 ** (nqubit - 1))
    for i in range(nqubit - 2, -1, -1):
        E = np.kron(E, pauli(j // (4 ** i)))
        j = j % (4 ** i)
    return E


def canonical_basis_ops(j, nqubit):
    """
    Computes the matrices of the canonical basis

    :param j: int between 0 and 4**nqubit-1
    :param nqubit: number of qubits
    :return: 2**nqubit x 2**nqubit array
    """
    d = 2 ** nqubit
    R = np.zeros((d, d), dtype='complex_')
    R[j // d, j % d] = 1
    return R


def krauss_repr_ops(m, rhoj, n, nqubit):
    # computes the Krauss representation of the operator given by ErhoE, where E are fixed set
    # of operators and rho is the one in canonical basis
    return np.dot(fixed_basis_ops(m, nqubit), np.dot(rhoj, np.conjugate(np.transpose(fixed_basis_ops(n, nqubit)))))


def thresh(X, eps=10 ** (-6)):
    """ Threshold function to cancel computational errors in the given input matrix
     """
    for i in range(len(X)):
        for j in range(len(X)):
            if np.abs(np.real(X[i, j])) < eps:
                X[i, j] -= np.real(X[i, j])
            if np.abs(np.imag(X[i, j])) < eps:
                X[i, j] -= 1j * np.imag(X[i, j])
    return X


class StatePreparationCircuit(Circuit):
    """
    Builds a preparation circuit to prepares a photon in each of the following
    logical Qubit state states: |0>,|1>,|+>,|+i>

    :param prep_state_basis_indxs: List of 'n'(=nqubit) elements indexing to choose jth
    state from any of the following Logical states: |0>,|1>,|+>,|+i>
    :param nqubit: Number of Qubits
    """

    def __init__(self, prep_state_basis_indxs: List, nqubit: int):
        super().__init__(m=2*nqubit, name="State Preparation Circuit")

        self._nqubit = nqubit
        self._prep_state_basis_indxs = prep_state_basis_indxs

    def _state_prep_circ_single_qubit(self, state_prep_circ_indx: int) -> Circuit:
        """
        Prepares a photon in any of the following Logical states: |0>,|1>,|+>,|+i>

        :param state_prep_circ_indx: int between 0 and 3 for states (|0>,|1>,|+>,|+i>)
        :return: 2 mode Preparation Circuit
        """
        assert 0 <= state_prep_circ_indx <= 3, f'Invalid index for circuit to prepare state'
        if state_prep_circ_indx == 1:
            return Circuit(2) // (0, PERM([1, 0]))
        elif state_prep_circ_indx == 2:
            return Circuit(2) // (0, BS.H())
        elif state_prep_circ_indx == 3:
            return Circuit(2) // (0, BS.H()) // (1, PS(np.pi / 2))
        else:
            return Circuit(2)

    def build_preparation_circuit(self):
        """
        Builds a circuit to prepare photons in chosen input basis for tomography experiment
        """
        for m in range(len(self._prep_state_basis_indxs)):
            self.add(2 * m, self._state_prep_circ_single_qubit(self._prep_state_basis_indxs[m]), merge=True)
        return self


class MeasurementCircuit(Circuit):
    """
    Builds a measurement circuit in the Pauli Basis (I,X,Y,Z) to perform tomography experiments.

    :param meas_basis_pauli_indxs: List of 'n'(=nqubit) elements indexing to choose jth
    Pauli Matrix {j:0=I, j:1=X, j:2=Y, j:3=Z} for measurement basis at nth Qubit
    [number of elements = between (0 -> 4**nqubits-1)]
    :param nqubit: Number of Qubits
    """

    def __init__(self, meas_basis_pauli_indxs: List, nqubit: int):
        super().__init__(m=2*nqubit, name="Measurement Basis Circuit")

        self._nqubit = nqubit
        self._meas_basis_pauli_indxs = meas_basis_pauli_indxs

    def _meas_circ_single_qubit(self, pauli_meas_circ_indx: int) -> Circuit:
        """
        Prepares 1 qubit circuits to measure a photon in the pauli basis I,X,Y,Z

        :param pauli_meas_circ_indx: int between 0 and 3
        :return: 2 modes Measurement Circuit
        """
        assert 0 <= pauli_meas_circ_indx <= 3, f'Invalid index for measurement circuit'

        if pauli_meas_circ_indx == 1:
            return Circuit(2) // (0, BS.H())
        elif pauli_meas_circ_indx == 2:
            return Circuit(2) // (0, BS.Rx(theta=np.pi / 2, phi_bl=np.pi, phi_br=-np.pi / 2))
        else:
            return Circuit(2)

    def build_measurement_circuit(self):
        """
        Builds the circuit to perform measurement of photons in the Pauli basis
        """
        for m in range(len(self._meas_basis_pauli_indxs)):
            self.add(2 * m, self._meas_circ_single_qubit(self._meas_basis_pauli_indxs[m]), merge=True)
        return self


class QuantumStateTomography(ATomography):
    def __init__(self, nqubit: int, operator_processor: Processor, post_process=False, renormalization=None):
        super().__init__(nqubit, operator_processor, post_process, renormalization)
        self._source = operator_processor.source  # default - ideal source
        self._backend = operator_processor.backend  # default - SLOSBackend()
        self._heralded_modes = [(key, value) for key, value in operator_processor.heralds.items()]


    @staticmethod
    def _list_subset_k_from_n(k, n):
        # list of distinct combination sets of length k from set 's' where 's' is the set {0,...,n-1}
        # used only in the method _stokes_parameter
        #  Should we put it in overall utils? or have a specific util for tomograph?
        s = {i for i in range(n)}
        return list(combinations(s, k))

    def _input_state_dist_config(self):
        input_state = BasicState("|1,0>")
        for _ in range(1, self._nqubit):
            # setting the input state for the gate qubit modes
            input_state *= BasicState("|1,0>")
        for m in self._heralded_modes:
            # setting the input for heralded modes of the given processor
            input_state *= BasicState([m[1]])
        input_distribution = self._source.generate_distribution(expected_input=input_state)
        return input_distribution

    def _tomography_processor(self, prep_state_basis_indxs, meas_basis_pauli_indxs):

        pc = StatePreparationCircuit(prep_state_basis_indxs, self._nqubit)  # state preparation circuit
        mc = MeasurementCircuit(meas_basis_pauli_indxs, self._nqubit)  # measurement basis circuit

        p = Processor(self._backend, self._nqubit*2, self._source)  # A Processor with corresponding backend and source
        qname = 'q'
        for i in range(self._nqubit):
            p.add_port(i*2, Port(Encoding.DUAL_RAIL, f'{qname}{i}'))  # set ports correctly

        p.add(0, pc.build_preparation_circuit())  # Add state preparation circuit to the left of the operator
        p.add(0, self._operator_processor)  # including the operator (as a processor)
        p.add(0, mc.build_measurement_circuit())  # Add measurement basis circuit to the right of the operator

        # Clear any inbuilt post-selection and heralding from perceval
        # - important for tomography to get output without inbuilt normalization of perceval

        p.clear_postselection()
        inbuilt_herald_ports = self._operator_processor.heralds  # to remove inbuilt heralds from Perceval processor
        for h_pos in inbuilt_herald_ports.keys():
            p.remove_port(h_pos)

        if self._post_process is True:
            # perhaps in future a post_process setup will be needed
            raise ValueError("Setting a postprocess is not implemented yet")

        if self._renormalization is None:
            # postselection on heralded modes if no renormalization
            ps = PostSelect()
            for m in self._heralded_modes:
                ps.eq([m[0]], m[1])
            p.set_postselection(ps)

        p.min_detected_photons_filter(0)  # QPU would have a problem with this - Eric
        p.with_input(self._input_state_dist_config())

        output_distribution = p.probs()["results"]
        return output_distribution

    def _stokes_parameter(self, prep_state_basis_indxs, meas_basis_pauli_indxs):
        """
        Computes the Stokes parameter S_i for state prep_state_basis_indxs after operator_circuit

        :param prep_state_basis_indxs: list of length of number of qubits representing the preparation circuit
        :param meas_basis_pauli_indxs: list of length of number of qubits representing the measurement circuit and the
         eigenvector we are measuring
        :return: float
        """
        output_distribution = self._tomography_processor(prep_state_basis_indxs, meas_basis_pauli_indxs)

        # calculation of the Stokes parameter begins here
        stokes_param = 0
        for k in range(self._nqubit + 1):
            for J in self._list_subset_k_from_n(k, self._nqubit):
                eta = 1
                if 0 not in J:
                    measurement_state = BasicState("|1,0>")
                else:
                    measurement_state = BasicState("|0,1>")
                    if meas_basis_pauli_indxs[0] != 0:
                        eta *= -1
                for j in range(1, self._nqubit):
                    if j not in J:
                        measurement_state *= BasicState("|1,0>")
                    else:
                        measurement_state *= BasicState("|0,1>")
                        if meas_basis_pauli_indxs[j] != 0:
                            eta *= -1
                for m in self._heralded_modes:
                    measurement_state *= BasicState([m[1]])
                stokes_param += eta * output_distribution[measurement_state]

        if self._renormalization is None:
            return stokes_param
        return stokes_param / self._renormalization

    def perform_quantum_state_tomography(self, state_index):
        """
        Computes the density matrix of a state after the operator_circuit

        :param state_index: list of length of number of qubits to index the corresponding preparation circuit
        :return: 2**nqubit x 2**nqubit array
        """
        d = 2 ** self._nqubit
        density_matrix = np.zeros((d, d), dtype='complex_')
        for j in range(d ** 2):
            i = [0] * self._nqubit
            j1 = j
            for k in range(self._nqubit - 1, -1, -1):
                i[k] = j1 // (4 ** k)
                j1 = j1 % (4 ** k)
            i.reverse()
            density_matrix += self._stokes_parameter(prep_state_basis_indxs=state_index, meas_basis_pauli_indxs=i) * \
                              fixed_basis_ops(j, self._nqubit)
        density_matrix = ((1 / 2) ** self._nqubit) * density_matrix
        return density_matrix

    def is_physical(self, density_matrix, eigen_tolerance=10 ** (-6)):
        """
        Verifies if chi matrix is trace preserving, hermitian, and completely positive (using the Choi matrix)

        :param chi_matrix: chi_matrix of a quantum map computed from Quantum Process Tomography
        :param eigen_tolerance: brings a tolerance for the positivity of the eigenvalues of the Choi matrix
        :return: list with findings of the tests
        """
        # Todo: density matrix is always hermitian and CP, trace maybe between 0 and 1 - output that
        # density matrix is CP, Hermitian, trace can be between 0 and 1

        res = super().is_physical(density_matrix, eigen_tolerance)
        d2 = len(density_matrix)

        # check if completely positive with Choi–Jamiołkowski isomorphism
        choi = 0
        for n in range(d2):
            P_n = np.conjugate(np.transpose(np.transpose([matrix_to_vector(np.transpose(fixed_basis_ops(n, self._nqubit)))])))
            for m in range(d2):
                choi += density_matrix[m, n] * np.dot(np.transpose([matrix_to_vector(np.transpose(fixed_basis_ops(m, self._nqubit)))]), P_n)
        choi /= 2 ** self._nqubit
        eigenvalues = np.linalg.eigvalsh(choi)
        if np.any(eigenvalues < -eigen_tolerance):
            val = np.round(eigenvalues[0], 5)
            res.append("|not Completely Positive|smallest eigenvalue :"+str(val))
        else:
            res.append("|Completely Positive|")

        return res


class QuantumProcessTomography(ATomography):
    def __init__(self, nqubit: int, operator_processor: Processor, post_process=False,
                 renormalization=None):
        super().__init__(nqubit, operator_processor, post_process, renormalization)
        self._qst = QuantumStateTomography(nqubit=self._nqubit, operator_processor=self._operator_processor,
                                           post_process=self._post_process, renormalization=self._renormalization)

    @staticmethod
    def _beta_ndarray(j, k, m, n, nqubit):
        d = 2 ** nqubit
        b = krauss_repr_ops(m, canonical_basis_ops(j, nqubit), n, nqubit)
        return b[k // d, k % d]

    def _beta_matrix(self):
        d = 2 ** self._nqubit
        M = np.zeros((d ** 4, d ** 4), dtype='complex_')
        for i in range(d ** 4):
            for j in range(d ** 4):
                M[i, j] = QuantumProcessTomography._beta_ndarray(i // (d ** 2), i % (d ** 2), j // (d ** 2),
                                                                 j % (d ** 2), self._nqubit)
        return M

    def _lambda_vector(self):
        """
        Computes the lambda vector of the operator

        :return: 2**(4*nqubit) vector
        """
        d = 2 ** self._nqubit
        EPS = []
        for state_counter in range(d ** 2):
            state_index = []  # indexes the preparation and measurement state from basis
            for i in range(self._nqubit - 1, -1, -1):
                state_index.append(state_counter // (4 ** i))
                state_counter = state_counter % (4 ** i)
            EPS.append(self._qst.perform_quantum_state_tomography(state_index))

        basis = matrix_basis(self._nqubit)
        L = np.zeros((d ** 2, d ** 2), dtype='complex_')
        for j in range(d ** 2):
            rhoj = canonical_basis_ops(j, self._nqubit)
            mu = decomp(rhoj, basis)
            eps_rhoj = sum([mu[i] * EPS[i] for i in range(d ** 2)])
            for k in range(d ** 2):
                L[j, k] = eps_rhoj[k // d, k % d]
        return matrix_to_vector(L)

    def _lambda_target(self, operator):
        # Implements a mathematical formula for ideal gate (given operator) to compute process fidelity
        d = 2 ** self._nqubit
        L = np.zeros((d ** 2, d ** 2), dtype='complex_')
        for j in range(d ** 2):
            rhoj = canonical_basis_ops(j, self._nqubit)
            eps_rhoj = np.linalg.multi_dot([operator, rhoj, np.conjugate(np.transpose(operator))])
            for k in range(d ** 2):
                L[j, k] = eps_rhoj[k // d, k % d]
        L1 = np.zeros((d ** 4, 1), dtype='complex_')
        for i in range(d ** 4):
            L1[i] = L[i // (d ** 2), i % (d ** 2)]
        return L1

    def chi_matrix(self):
        """
        Computes the chi matrix of the operator_circuit

        :return: 2**(2*nqubit)x2**(2*nqubit) array
        """
        Binv = np.linalg.pinv(self._beta_matrix())
        L = self._lambda_vector()
        X = np.dot(Binv, L)
        return vector_to_matrix(X)

    def chi_target(self, operator):
        # Implements a mathematical formula for ideal gate (given operator) to compute process fidelity
        beta = self._beta_matrix()
        lambd = self._lambda_target(operator)
        X = np.dot(np.linalg.pinv(beta), lambd)
        return vector_to_matrix(X)

    def process_fidelity(self, chi_computed, chi_ideal):
        """
        Computes the process fidelity of an operator and its perceval circuit

        :param chi_computed: chi matrix computed from process tomography of shape [2**(2*nqubit) x 2**(2*nqubit)]
        :param chi_ideal: Ideal chi matrix for the corresponding operator
        :return: float between 0 and 1
        """
        return np.real(np.trace(np.dot(chi_computed, chi_ideal)))

    def average_fidelity(self, operator):
        """
        Computes the average fidelity of an operator and its perceval circuit

        :param qst: QuantumStateTomgraphy object on which tomography is performed
        :return: float between 0 and 1
        """
        Udag = np.transpose(np.conjugate(operator))
        d = 2 ** self._nqubit
        f = 1 / (d + 1)

        # compute the map on a basis of states (tensor products of |0>, |1>, |+>,|i+>)
        EPS = []
        for state_counter in range(d ** 2):
            state_index = []
            for i in range(self._nqubit - 1, -1, -1):
                state_index.append(state_counter // (4 ** i))
                state_counter = state_counter % (4 ** i)
            EPS.append(self._qst.perform_quantum_state_tomography(state_index))

        basis = matrix_basis(self._nqubit)
        for j in range(d ** 2):
            Uj = fixed_basis_ops(j, self._nqubit)
            mu = decomp(Uj, basis)
            eps_Uj = sum([mu[i] * EPS[i] for i in range(d ** 2)])  # compute the map on a basis
            Ujdag = np.transpose(np.conjugate(Uj))
            a = np.linalg.multi_dot([operator, Ujdag, Udag, eps_Uj])
            f += (1 / ((d + 1) * (d ** 2))) * np.trace(a)
        return np.real(f)

    def error_process_matrix(self, computed_chi, operator):
        """
        Computes the error matrix for an operation from the computed chi matrix

        :param computed_chi:
        :return: matrix
        """
        d = 2 ** self._nqubit
        V = np.zeros((d ** 2, d ** 2), dtype='complex_')
        for m in range(d ** 2):
            for n in range(d ** 2):
                Emdag = np.transpose(np.conjugate(fixed_basis_ops(m, self._nqubit)))
                En = fixed_basis_ops(n, self._nqubit)
                Udag = np.transpose(np.conjugate(operator))
                V[m, n] = (1 / d) * np.trace(np.linalg.multi_dot([Emdag, En, Udag]))
        return np.linalg.multi_dot([V, computed_chi, np.conjugate(np.transpose(V))])

    def is_physical(self, chi_matrix, eigen_tolerance=10 ** (-6)):
        """
        Verifies if chi matrix is trace preserving, hermitian, and completely positive (using the Choi matrix)

        :param chi_matrix: chi_matrix of a quantum map computed from Quantum Process Tomography
        :param eigen_tolerance: brings a tolerance for the positivity of the eigenvalues of the Choi matrix
        :return: list with findings of the tests
        """
        res = super().is_physical(chi_matrix, eigen_tolerance)
        d2 = len(chi_matrix)

        # check if completely positive with Choi–Jamiołkowski isomorphism
        choi = 0
        for n in range(d2):
            P_n = np.conjugate(np.transpose(np.transpose([matrix_to_vector(np.transpose(fixed_basis_ops(n, self._nqubit)))])))
            for m in range(d2):
                choi += chi_matrix[m, n] * np.dot(np.transpose([matrix_to_vector(np.transpose(fixed_basis_ops(m, self._nqubit)))]), P_n)
        choi /= 2 ** self._nqubit
        eigenvalues = np.linalg.eigvalsh(choi)
        if np.any(eigenvalues < -eigen_tolerance):
            val = np.round(eigenvalues[0], 5)
            res.append("|not Completely Positive|smallest eigenvalue :"+str(val))
        else:
            res.append("|Completely Positive|")

        return res

# todo: lack of documentation about converter in tools on perceval documentation. verify wbefore the next version
#  release
