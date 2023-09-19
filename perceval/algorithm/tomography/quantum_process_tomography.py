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
import perceval as pcvl
from perceval.components import BS, PS, PERM, Unitary
from perceval.components.source import Source
from perceval.components import Circuit
import itertools
from scipy.stats import unitary_group
from perceval.simulators import Simulator
from perceval.backends import SLOSBackend
from typing import List
from _tomography_utils import state_to_dens_matrix, compute_matrix, matrix_basis, matrix_to_vector, \
    vector_to_matrix, decomp


def pauli(j):
    """
    computes the j-th Pauli operator (I,X,Y,Z)

    :param j: int between 0 and 3
    :return: 2x2 unitary and hermitian array
    """
    assert j < 3, f'Pauli Index should be less than 3'
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
    :return: 2**nqubitx2**nqubit array
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
    :return: 2**nqubitx2**nqubit array
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
    # todo: function not in use, why here?
    """ Threshold function to cancel computational errors in $\chi$
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
            return pcvl.Circuit(2) // (0, PERM([1, 0]))
        elif state_prep_circ_indx == 2:
            return pcvl.Circuit(2) // (0, BS.H())
        elif state_prep_circ_indx == 3:
            return pcvl.Circuit(2) // (0, BS.H()) // (1, PS(np.pi / 2))
        else:
            return pcvl.Circuit(2)

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
            return pcvl.Circuit(2) // (0, BS.H())
        elif pauli_meas_circ_indx == 2:
            return pcvl.Circuit(2) // (0, BS.Rx(theta=np.pi / 2, phi_bl=np.pi, phi_br=-np.pi / 2))
        else:
            return pcvl.Circuit(2)

    def build_measurement_circuit(self):
        """
        Builds the circuit to perform measurement of photons in the Pauli basis
        """
        for m in range(len(self._meas_basis_pauli_indxs)):
            self.add(2 * m, self._meas_circ_single_qubit(self._meas_basis_pauli_indxs[m]), merge=True)
        return self


class QuantumStateTomography:
    def __init__(self, operator: Circuit, nqubit: int, source: Source = Source(), backend=SLOSBackend(),
                 heralded_modes: List = [], post_process=False, renormalization=None):
        self._source = source  # default - ideal source
        self._backend = backend  # default - SLOSBackend()
        self._operator = operator
        self._nqubit = nqubit
        self._post_process = post_process
        self._renormalization = renormalization
        self._heralded_modes = heralded_modes

    def _tomography_circuit(self, num_state: List, i: List) -> Circuit:
        # todo: rename/refactor "num_state" and "i" parameter
        tomography_circuit = pcvl.Circuit(2 * self._nqubit + len(self._heralded_modes))

        pc = StatePreparationCircuit(num_state, self._nqubit)  # state preparation
        tomography_circuit.add(0, pc.build_preparation_circuit())

        tomography_circuit.add(0, self._operator)  # unknown operator

        mc = MeasurementCircuit(i, self._nqubit)  # measurement operator
        tomography_circuit.add(0, mc.build_measurement_circuit())

        return tomography_circuit

    def _list_subset_k_from_n(self, k, n):
        # list of distinct combination sets of length k from set 's' where 's' is the set {0,...,n-1}
        # used only in the method _stokes_parameter
        #  Should we put it in overall utils? or have a specific util for tomograph?
        s = {i for i in range(n)}
        return list(itertools.combinations(s, k))

    def _stokes_parameter(self, num_state, i):
        """
        Computes the Stokes parameter S_i for state num_state after operator_circuit

        :param num_state: list of length of number of qubits representing the preparation circuit
        :param i: list of length of number of qubits representing the measurement circuit and the eigenvector we are
        measuring
        :return: float
        """
        tomography_circ = self._tomography_circuit(num_state, i)

        simulator = Simulator(self._backend)  # todo: change to Processor
        simulator.set_circuit(tomography_circ)

        if self._renormalization is None:  # postselection if no renormalization
            ps = pcvl.PostSelect()
            if self._post_process:
                for m in range(self._nqubit):
                    ps.eq([2 * m, 2 * m + 1], 1)
            for m in self._heralded_modes:
                ps.eq([m[0]], m[1])
            simulator.set_postselection(ps)

        input_state = pcvl.BasicState("|1,0>")  # input state accounting the heralded modes
        for _ in range(1, self._nqubit):
            input_state *= pcvl.BasicState("|1,0>")
        for m in self._heralded_modes:
            input_state *= pcvl.BasicState([m[1]])
        input_distribution = self._source.generate_distribution(expected_input=input_state)

        simulator.set_min_detected_photon_filter(0)
        output_distribution = simulator.probs_svd(input_distribution)["results"]

        # calculation of the Stokes parameter begins here
        stokes_param = 0
        for k in range(self._nqubit + 1):
            for J in self._list_subset_k_from_n(k, self._nqubit):
                eta = 1
                if 0 not in J:
                    measurement_state = pcvl.BasicState("|1,0>")
                else:
                    measurement_state = pcvl.BasicState("|0,1>")
                    if i[0] != 0:
                        eta *= -1
                for j in range(1, self._nqubit):
                    if j not in J:
                        measurement_state *= pcvl.BasicState("|1,0>")
                    else:
                        measurement_state *= pcvl.BasicState("|0,1>")
                        if i[j] != 0:
                            eta *= -1
                for m in self._heralded_modes:
                    measurement_state *= pcvl.BasicState([m[1]])
                stokes_param += eta * output_distribution[measurement_state]

        if self._renormalization is None:
            return stokes_param
        return stokes_param / self._renormalization

    def perform_quantum_state_tomography(self, state):
        """
        Computes the density matrix of a state after the operator_circuit

        :param state: list of length of number of qubits representing the preparation circuit
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
            density_matrix += self._stokes_parameter(state, i) * fixed_basis_ops(j, self._nqubit)
        density_matrix = ((1 / 2) ** self._nqubit) * density_matrix
        return density_matrix


class QuantumProcessTomography:
    # TODO: what if i pass QST object here? how are parameters handled. what i want to do is create a obj for QST,
    #  perform that, then ask to do QPT, then fidelity with that.
    def __init__(self, nqubit: int, operator: Circuit, density_matrix: np.ndarray, heralded_modes=[],
                 post_process=False, renormalization=None):
        self._nqubit = nqubit
        self._operator = operator
        self._density_matrix = density_matrix
        self._heralded_modes = heralded_modes
        self._post_process = post_process
        self._renormalization = renormalization

    @staticmethod
    def _beta_mult_qubit(j, k, m, n, nqubit):
        d = 2 ** nqubit
        b = krauss_repr_ops(m, canonical_basis_ops(j, nqubit), n, nqubit)
        return b[k // d, k % d]

    def _beta_matrix_mult_qubit(self):
        d = 2 ** self._nqubit
        M = np.zeros((d ** 4, d ** 4), dtype='complex_')
        for i in range(d ** 4):
            for j in range(d ** 4):
                M[i, j] = QuantumProcessTomography._beta_mult_qubit(i // (d ** 2), i % (d ** 2),
                                                                    j // (d ** 2), j % (d ** 2), self._nqubit)
        return M

    def _lambda_mult_qubit(self):
        """
        Computes the lambda vector of the operator

        :return: 2**(4*nqubit) vector
        """
        d = 2 ** self._nqubit
        EPS = []
        for state in range(d ** 2):
            l = []
            for i in range(self._nqubit - 1, -1, -1):
                l.append(state // (4 ** i))
                state = state % (4 ** i)
            EPS.append(self._density_matrix)
        basis = matrix_basis(self._nqubit)
        L = np.zeros((d ** 2, d ** 2), dtype='complex_')
        for j in range(d ** 2):
            rhoj = canonical_basis_ops(j, self._nqubit)
            mu = decomp(rhoj, basis)
            eps_rhoj = sum([mu[i] * EPS[i] for i in range(d ** 2)])
            for k in range(d ** 2):
                L[j, k] = eps_rhoj[k // d, k % d]
        return matrix_to_vector(L)

    def _lambda_mult_ideal(self):
        # Implements a mathematical formula for ideal gate (given operator) to compute process fidelity
        d = 2 ** self._nqubit
        L = np.zeros((d ** 2, d ** 2), dtype='complex_')
        for j in range(d ** 2):
            rhoj = canonical_basis_ops(j, self._nqubit)
            eps_rhoj = np.linalg.multi_dot([self._operator, rhoj, np.conjugate(np.transpose(self._operator))])
            for k in range(d ** 2):
                L[j, k] = eps_rhoj[k // d, k % d]
        L1 = np.zeros((d ** 4, 1), dtype='complex_')
        for i in range(d ** 4):
            L1[i] = L[i // (d ** 2), i % (d ** 2)]
        return L1

    def chi_mult_qubit(self):
        """
        Computes the chi matrix of the operator

        :return: 2**(2*nqubit)x2**(2*nqubit) array
        """
        Binv = np.linalg.pinv(self._beta_matrix_mult_qubit())
        L = self._lambda_mult_qubit()
        X = np.dot(Binv, L)
        return vector_to_matrix(X)

    def chi_mult_ideal(self):
        # Implements a mathematical formula for ideal gate (given operator) to compute process fidelity
        X = np.dot(np.linalg.pinv(self._beta_matrix_mult_qubit()), self._lambda_mult_ideal())
        return vector_to_matrix(X)


class FidelityTomography:
    def __init__(self, qpt: QuantumProcessTomography, operator, nqubit):
        self._qpt = qpt  # Default source => brightness=1, g2=0, indistinguishability=1, loss=0
        self._operator = operator
        self._nqubit = nqubit
        print("I am going to compute fidelity based on tomography process")

    def process_fidelity(self, chi_computed):
        """
        Computes the process fidelity of an operator and its perceval circuit

        :param chi_computed: chi matrix computed from process tomography of shape [2**(2*nqubit) x 2**(2*nqubit)]
        :return: float between 0 and 1
        """
        X0 = chi_computed # qpt.chi_mult_qubit(operator_circuit, nqubit, heralded_modes, post_process, renormalization)
        X1 = QuantumProcessTomography.chi_mult_ideal(self._operator, self._nqubit) # todo: fix params
        return np.real(np.trace(np.dot(X0, X1)))

    def random_fidelity(self, nqubit):
        """
        Computes the process and the average fidelity of a random non-entangling operator
        Does not take any input from user; create a random circuit and tests fidelity
        #todo: perhaps use it in a test

        :param nqubit: number of qubits
        """
        L = []
        for i in range(nqubit):
            L.append(unitary_group.rvs(2))
        print('test matrices :', L)
        M = L[0]
        CU = pcvl.Circuit(2 * nqubit).add(0, Unitary(pcvl.Matrix(L[0])))
        for i in range(1, nqubit):
            M = np.kron(M, L[i])
            CU.add(2 * i, Unitary(pcvl.Matrix(L[i])))
        pf = self.process_fidelity(M, CU)
        afq = self.average_fidelity(M, CU)
        print('process fidelity :', pf, '\n',
              'average fidelity :', afq)

    def average_fidelity_with_reconstruction(self, operator, operator_circuit, heralded_modes=[], post_process=False,
                                             renormalization=None):
        """
        not so important-computes avg fideltiy in a longer way
        Computes the average fidelity of an operator and its perceval circuit by recontruction of the whole map

        :param operator: matrix for the operator
        :param operator_circuit: perceval circuit for the operator
        :param heralded_modes: list of tuples giving for each heralded mode the number of heralded photons
        :param post_process: bool for postselection on the outcome or not
        :param renormalization: float (success probability of the gate) by which we renormalize the map instead of just
        doing postselection which to non CP maps
        :return: float between 0 and 1
        """
        nqubit = int(np.log2(len(operator)))
        Udag = np.transpose(np.conjugate(operator))
        d = 2 ** nqubit
        f = 1 / (d + 1)
        for j in range(d ** 2):
            Uj = QuantumProcessTomography.E(j, nqubit)
            Ujdag = np.transpose(np.conjugate(Uj))
            eps_Uj = self.map_reconstructed(Uj, operator_circuit, nqubit, heralded_modes, post_process, renormalization)
            a = np.linalg.multi_dot([operator, Ujdag, Udag, eps_Uj])
            f += (1 / ((d + 1) * (d ** 2))) * np.trace(a)
        return np.real(f)

    def error_process_matrix(self, operator, operator_circuit, heralded_modes=[], post_process=False,
                             renormalization=None):
        """
        Computes the error matrix for an operation from the chi matrix

        :param operator: matrix for the operator
        :param operator_circuit: perceval circuit for the operator
        :param heralded_modes: list of tuples giving for each heralded mode the number of heralded photons
        :param post_process: bool for postselection on the outcome or not
        :param renormalization: float (success probability of the gate) by which we renormalize the map instead of just
        doing postselection which to non CP maps
            :param brightness source brightness
        :param g2 SPS g2
        :param indistinguishability photon indistinguishability
        :param loss known losses in source
        :return: matrix
        """
        nqubit = int(np.log2(len(operator)))
        d = 2 ** nqubit
        qpt = QuantumProcessTomography()
        X = qpt.chi_mult_qubit(operator_circuit, nqubit, heralded_modes, post_process, renormalization)
        V = np.zeros((d ** 2, d ** 2), dtype='complex_')
        for m in range(d ** 2):
            for n in range(d ** 2):
                Emdag = np.transpose(np.conjugate(QuantumProcessTomography.E(m, nqubit)))
                En = QuantumProcessTomography.E(n, nqubit)
                Udag = np.transpose(np.conjugate(operator))
                V[m, n] = (1 / d) * np.trace(np.linalg.multi_dot([Emdag, En, Udag]))
        return np.linalg.multi_dot([V, X, np.conjugate(np.transpose(V))])

    def average_fidelity(self, operator, operator_circuit, heralded_modes=[], post_process=False, renormalization=None):
        """
        Computes the average fidelity of an operator and its perceval circuit

        :param operator: matrix for the operator
        :param operator_circuit: perceval circuit for the operator
        :param heralded_modes: list of tuples giving for each heralded mode the number of heralded photons
        :param post_process: bool for postselection on the outcome or not
        :param renormalization: float (success probability of the gate) by which we renormalize the map instead of just
        doing postselection which to non CP maps
            :param brightness source brightness
        :param g2 SPS g2
        :param indistinguishability photon indistinguishability
        :param loss known losses in source
        :return: float between 0 and 1
        """
        nqubit = int(np.log2(len(operator)))
        Udag = np.transpose(np.conjugate(operator))
        d = 2 ** nqubit
        f = 1 / (d + 1)

        # compute the map on a basis of states (tensor products of |0>, |1>, |+>,|i+>)
        EPS = []
        for state in range(d ** 2):
            l = []
            for i in range(nqubit - 1, -1, -1):
                l.append(state // (4 ** i))
                state = state % (4 ** i)
            qst = QuantumStateTomography()
            # todo: fix instance params
            EPS.append(qst.perform_quantum_state_tomography(l, operator_circuit, nqubit, heralded_modes, post_process,
                                                            renormalization))

        basis = matrix_basis(nqubit)
        for j in range(d ** 2):
            Uj = QuantumProcessTomography.E(j, nqubit)
            mu = decomp(Uj, basis)
            eps_Uj = sum([mu[i] * EPS[i] for i in range(d ** 2)])  # compute the map on a basis
            Ujdag = np.transpose(np.conjugate(Uj))
            a = np.linalg.multi_dot([operator, Ujdag, Udag, eps_Uj])
            f += (1 / ((d + 1) * (d ** 2))) * np.trace(a)
        return np.real(f)

    def mixture(self, operator_circuit, nqubit, heralded_modes=[], post_process=False, renormalization=None):
        """
        ## for CNOT gate - not so important
        Computes the mixture created by a perceval circuit

        :param operator_circuit: perceval circuit for the operator
        :param nqubit: number of qubits
        :param heralded_modes: list of tuples giving for each heralded mode the number of heralded photons
        :param post_process: bool for postselection on the outcome or not
        :param renormalization: float (success probability of the gate) by which we renormalize the map instead of just
        doing postselection which to non CP maps
        :param brightness source brightness
        :param g2 SPS g2
        :param indistinguishability photon indistinguishability
        :param loss known losses in source

        :return: float between 0 and 1
        """
        d = 2 ** nqubit
        qpt = QuantumProcessTomography()
        X = qpt.chi_mult_qubit(operator_circuit, nqubit, heralded_modes, post_process, renormalization)
        t = np.trace(np.dot(X, X))
        return np.real((d * t + 1) / (d + 1))

    def is_physical(matrix, eigen_tolerance=10 ** (-6)):
        """
        Verifies if a matrix is trace preserving, hermitian, and completely positive (using the Choi matrix)

        :param matrix: square matrix
        :param eigen_tolerance: brings a tolerance for the positivity of the eigenvalues of the Choi matrix
        :return: bool and string
        """
        # check if trace preserving
        b = True
        s = ""
        if not np.isclose(np.trace(matrix), 1):
            b = False
            print("trace :", np.trace(matrix))
            s += "|trace not 1|"
        # check if hermitian
        n = len(matrix)
        for i in range(n):
            for j in range(i, n):
                if not np.isclose(matrix[i][j], np.conjugate(matrix[j][i])):
                    b = False
                    s += "|not hermitian|"
        # check if completely positive with Choi–Jamiołkowski isomorphism
        M = np.kron(matrix, np.eye((n), dtype='complex_'))
        I = np.eye((n), dtype='complex_')
        omega = sum([np.kron(I[:, i], I[:, i]) for i in range(n)])
        rho = np.dot(omega, np.conjugate(np.transpose(omega)))
        choi = np.dot(M, rho)
        eigenvalues = np.linalg.eigvalsh(choi)
        if np.any(eigenvalues < -eigen_tolerance):
            b = False
            print("smallest eigenvalue :", eigenvalues[0])
            s += "|not CP|"
        if b:
            return True
        return False, s

    def map_reconstructed(self, rho, operator_circuit, nqubit, heralded_modes=[], post_process=False,
                          renormalization=None):
        """
        Computes the image of a density matrix by the operator using the chi matrix

        :param rho: any density matrix
        :param operator_circuit: perceval circuit for the operator
        :param nqubit: number of qubits
        :param heralded_modes: list of tuples giving for each heralded mode the number of heralded photons
        :param post_process: bool for postselection on the outcome or not
        :param renormalization: float (success probability of the gate) by which we renormalize the map instead of just
        doing postselection which to non CP maps
            :param brightness source brightness
        :param g2 SPS g2
        :param indistinguishability photon indistinguishability
        :param loss known losses in source

        :return: float between 0 and 1
        """
        d = 2 ** nqubit
        qpt = QuantumProcessTomography()
        X = qpt.chi_mult_qubit(operator_circuit, nqubit, heralded_modes, post_process, renormalization)
        eps = np.zeros((d, d), dtype='complex_')
        for m in range(d ** 2):
            for n in range(d ** 2):
                eps += X[m, n] * np.linalg.multi_dot([QuantumProcessTomography.E(m, nqubit), rho, np.transpose(np.conjugate(QuantumProcessTomography.E(n, nqubit)))])
        # Eqn 2.4 the exact sum
        return eps
