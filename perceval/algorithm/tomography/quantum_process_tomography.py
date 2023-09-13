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
from perceval.components import catalog, BS, PS, PERM, Unitary
from perceval.components.source import Source
from perceval.components import ACircuit, Circuit
import itertools
from scipy.stats import unitary_group
from perceval.simulators import Simulator
from perceval.backends import SLOSBackend
from typing import List


# ##### threshold ######################################################################################
def thresh(X, eps=10 ** (-6)):
    """ Threshold function to cancel computational errors in $\chi$
     """
    for i in range(len(X)):
        for j in range(len(X)):
            if np.abs(np.real(X[i, j])) < eps:
                X[i, j] -= np.real(X[i, j])
            if np.abs(np.imag(X[i, j])) < eps:
                X[i, j] -= 1j * np.imag(X[i, j])
    return X


def pauli(j):
    """
    computes the j-th Pauli operator (I,X,Y,Z)

    :param j: int between 0 and 3
    :return: 2x2 unitary and hermitian array
    """
    if j > 3:
        raise TypeError
    else:
        if j == 0:  # I
            return np.eye(2, dtype='complex_')
        elif j == 1:  # Pauli X
            return np.array([[0, 1], [1, 0]], dtype='complex_')
        elif j == 2:  # Pauli Y
            return np.array([[0, -1j], [1j, 0]], dtype='complex_')
        else:  # Pauli Z
            return np.array([[1, 0], [0, -1]], dtype='complex_')


def E(j, nqubit):
    """
    computes the fixed sets of operators (tensor products of pauli gates) for quantum process tomography

    :param j: int between 0 and 4**nqubit-1
    :param nqubit: number of qubits
    :return: 2**nqubitx2**nqubit array
    """
    if nqubit == 1:
        return pauli(j)

    E = pauli(j // (4 ** (nqubit - 1)))
    j = j % (4 ** (nqubit - 1))  # todo: ask about how values of "j" and how they are modified here
    for i in range(nqubit - 2, -1, -1):
        E = np.kron(E, pauli(j // (4 ** i)))
        j = j % (4 ** i)
    return E


def rho(j, nqubit):  # canonical basis
    """
    Computes the matrices of the canonical basis

    :param j: int between 0 and 4**nqubit-1
    :param nqubit: number of qubits
    :return: 2**nqubitx2**nqubit array
    """
    d = 2 ** nqubit
    R = np.zeros((d, d), dtype='complex_')
    R[j // d, j % d] = 1
    return R  # todo: ask Arman to confirm which rho is this


def ErhoE(m, rhoj, n, nqubit):
    return np.dot(E(m, nqubit), np.dot(rhoj, np.conjugate(np.transpose(E(n, nqubit)))))


# ##### preparation circuit ######################################################################################
class PreparationCircuit:
    """
    Builds a preparation circuit to prepares one photon in each of the following states: |1,0>,|0,1>,|+>,|+i>

    :param j: jth Pauli Matrix I,X,Y,Z. Value of j between 0 and 3
    :param nqubit: Number of Qubits
    """

    def __init__(self, j: List, nqubit: int) -> Circuit:
        # todo: fix input name for j and its documentation
        self._nqubit = nqubit
        self._j = j
        self._prep_circuit = Circuit(2 * nqubit, name="Preparation Circ")

    def _prep_circ_single_qubit(self, some_name: int) -> Circuit: #todo: Arman - can you help me choose name?
        """
        Prepares one photon in each of the following states: |1,0>,|0,1>,|+>,|+i>

        :param some_name: int between 0 and 3 #todo: SOME NAME DOESNT HELP OR DO ANYTHING FOR SINGLE QUBIT :-(
        #todo: need to keep j as list for both single/multi but maybe assert here that it is always single element?
        :return: 2 modes perceval circuit
        """
        print("I AM HERE AND MY j is", some_name)
        if some_name[0] == 1:
            print("i CAN NOW REACH here")
            return self._prep_circuit.add(0, PERM([1, 0]))
        if some_name == 2:
            return self._prep_circuit.add(0, BS.H())
        if some_name == 3:
            return self._prep_circuit.add(0, BS.H()).add(1, PS(np.pi / 2))

    def _prep_circ_multi_qubit(self, j: List):
        """
        Prepares each photon in each of the following states: |1,0>,|0,1>,|+>,|+i>

        :param j: List of int between 0 and 4**nqubit-1 todo: verify
        :param nqubit: number of qubits
        :return: 2*nqubit modes perceval circuit
        """
        for m in range(len(j)):
            #todo do they neede to be added or worked in what manner
            return self._prep_circuit.add(2 * m, self._prep_circ_single_qubit(some_name=j[m]), merge=True)

    def build_preparation_circuit(self):
        if self._nqubit == 1:
            self._prep_circ_single_qubit(self._j)
        else:
            self._prep_circ_multi_qubit(self._j)
        return self._prep_circuit


# ##### measurement circuit ######################################################################################
class MeasurementCircuit:
    """
    # todo: fix input name for j and its documentation
    Builds a measurement circuit in the Pauli Basis (I,X,Y,Z) to perform tomography experiments.

    :param j: jth Pauli Matrix I,X,Y,Z. Value of j between 0 and 3
    :param nqubit: Number of Qubits
    """

    def __init__(self, j: List, nqubit: int) -> Circuit:
        # todo: fix input type for j and its documentation
        self._nqubit = nqubit
        self._j = j
        self._meas_circuit = Circuit(2 * nqubit, name="Measurement Circ")

    def _meas_circ_single_qubit(self, some_other_name: int) -> Circuit:
        #todo: Arman could you help me fogure out name?
        """
        Measures the photon in the pauli basis I,X,Y,Z

        :param some_other_name: int between 0 and 3
        :return: 2 modes perceval circuit
        """
        if some_other_name == 1:
            return self._meas_circuit.add(0, BS.H())
        elif some_other_name == 2:
            return self._meas_circuit.add(0, BS.Rx(theta=np.pi / 2, phi_bl=np.pi, phi_br=-np.pi / 2))
        else:
            return self._meas_circuit

    def _meas_circ_multi_qubit(self, j: List):
        """
         Measures each photon in the pauli basis

        :param j: int between 0 and 4**nqubit-1
        :param nqubit: number of qubits
        :return: 2*nqubit modes perceval circuit
        """
        for m in range(len(j)):
            return self._meas_circuit.add(2 * m, self.meas_circ_single_qubit(some_other_name=j[m]), merge=True)

    def build_measurement_circuit(self):
        if self._nqubit == 1:
            self._meas_circ_single_qubit(self._j)
        else:
            self._meas_circ_multi_qubit(self._j)
        return self._meas_circuit


# ##### P and Stokes are part of QST ##############################################################################

class QuantumStateTomography:
    def __init__(self):
        pass

    def tomography_circuit(self, num_state: List, i: List, heralded_modes: List, nqubit: int,
                           operator_circuit: Circuit) -> Circuit:
        tomography_circuit = pcvl.Circuit(2 * nqubit + len(heralded_modes))
        # state preparation
        pc = PreparationCircuit(num_state, nqubit)
        tomography_circuit.add(0, pc.build_preparation_circuit())
        # unknown operator
        tomography_circuit.add(0, operator_circuit)
        # measurement operator
        mc = MeasurementCircuit(i, nqubit)
        tomography_circuit.add(0, mc.build_measurement_circuit())
        return tomography_circuit

    def probs_finding_state_kth_qbit(self, k, n):
        # todo: MISNOMER; not a probability - it simply is forming nCk or C(n,k) terms whose products are summed
        #  or something - see equation in ntoes again and decide
        # set of subsets of size k in {0,...,n-1}
        s = {i for i in range(n)}
        return list(itertools.combinations(s, k))

    def stokes_parameter(self, num_state, operator_circuit, i, heralded_modes=[], post_process=False,
                         renormalization=None, brightness=1, g2=0, indistinguishability=1, loss=0):
        """
        Computes the Stokes parameter S_i for state num_state after operator_circuit

        :param num_state: list of length of number of qubits representing the preparation circuit todo: why a list?
        :param operator_circuit: perceval circuit for the operator
        :param i: list of length of number of qubits representing the measurement circuit and the eigenvector we are measuring
        :param heralded_modes: list of tuples giving for each heralded mode the number of heralded photons
        :param post_process: bool for postselection on the outcome or not
        :param renormalization: float (success probability of the gate) by which we renormalize the map instead of just
        doing postselection which to non CP maps
        :param brightness source brightness
        :param g2 SPS g2
        :param indistinguishability photon indistinguishability
        :param loss known losses in source
        :return: float
        """
        nqubit = len(i)
        # todo: Arman doubt: i can be different than num_state, nqubit is always with i?
        # QPT CIRCUIT : TODO: what is this circuit supposed to look like?
        qpt_circuit = self.tomography_circuit(num_state, i, heralded_modes, nqubit, operator_circuit)

        source = Source(emission_probability=brightness, multiphoton_component=g2,
                        indistinguishability=indistinguishability, losses=loss)
        # todo: remove source parameters from each function and simply pass Source with set params in code.

        simulator = Simulator(SLOSBackend())  # todo: Arman do we need to always use SLOS? or user can choose?
        simulator.set_circuit(qpt_circuit)

        if renormalization is None:  # postselection if no renormalization
            ps = pcvl.PostSelect()
            if post_process:
                for m in range(nqubit):
                    ps.eq([2 * m, 2 * m + 1], 1)
            for m in heralded_modes:
                ps.eq([m[0]], m[1])
            simulator.set_postselection(ps)

        input_state = pcvl.BasicState("|1,0>")  # input state accounting the heralded modes
        for _ in range(1, nqubit):
            input_state *= pcvl.BasicState("|1,0>")
        for m in heralded_modes:
            input_state *= pcvl.BasicState([m[1]])
        input_distribution = source.generate_distribution(expected_input=input_state)

        simulator.set_min_detected_photon_filter(0)
        output_distribution = simulator.probs_svd(input_distribution)["results"]

        stokes_param = 0  # calculation of the Stokes parameter begins here
        for k in range(nqubit + 1):
            for J in self.probs_finding_state_kth_qbit(k, nqubit):
                eta = 1
                if 0 not in J:
                    measurement_state = pcvl.BasicState("|1,0>")
                else:
                    measurement_state = pcvl.BasicState("|0,1>")
                    if i[0] != 0:
                        eta *= -1
                for j in range(1, nqubit):
                    if j not in J:
                        measurement_state *= pcvl.BasicState("|1,0>")
                    else:
                        measurement_state *= pcvl.BasicState("|0,1>")
                        if i[j] != 0:
                            eta *= -1
                for m in heralded_modes:
                    measurement_state *= pcvl.BasicState([m[1]])
                stokes_param += eta * output_distribution[measurement_state]

        if renormalization is None:
            return stokes_param
        return stokes_param / renormalization

    def perform_quantum_state_tomography(self, state, operator_circuit, nqubit, heralded_modes=[], post_process=False,
                                 renormalization=None, brightness=1, g2=0, indistinguishability=1, loss=0):
        """
        Computes the density matrix of a state after the operator_circuit

        :param state: list of length of number of qubits representing the preparation circuit
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
        :return: 2**nqubitx2**nqubit array
        """
        d = 2 ** nqubit
        density_matrix = np.zeros((d, d), dtype='complex_')
        for j in range(d ** 2):
            i = [0] * nqubit
            j1 = j
            for k in range(nqubit - 1, -1, -1):
                i[k] = j1 // (4 ** k)
                j1 = j1 % (4 ** k)
            i.reverse()
            density_matrix += self.stokes_parameter(state, operator_circuit, i, heralded_modes, post_process,
                                               renormalization,
                                               brightness, g2, indistinguishability, loss) * E(j, nqubit)
        density_matrix = ((1 / 2) ** nqubit) * density_matrix
        return density_matrix


# ##### IDK ######################################################################################
def beta_mult_qubit(j, k, m, n, nqubit):
    d = 2 ** nqubit
    b = ErhoE(m, rho(j, nqubit), n, nqubit)
    return b[k // d, k % d]


def beta_matrix_mult_qubit(nqubit):
    d = 2 ** nqubit
    M = np.zeros((d ** 4, d ** 4), dtype='complex_')
    for i in range(d ** 4):
        for j in range(d ** 4):
            M[i, j] = beta_mult_qubit(i // (d ** 2), i % (d ** 2), j // (d ** 2), j % (d ** 2), nqubit)
    return M


def lambd_mult_qubit(operator_circuit, nqubit, heralded_modes=[], post_process=False, renormalization=None,
                     brightness=1, g2=0, indistinguishability=1, loss=0):
    """
    Computes the lambda vector of the operator

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
    :return: 2**(4*nqubit) vector
    """
    d = 2 ** nqubit
    EPS = []
    for state in range(d ** 2):
        l = []
        for i in range(nqubit - 1, -1, -1):
            l.append(state // (4 ** i))
            state = state % (4 ** i)
        qst = QuantumStateTomography()
        # todo: fix instance params
        EPS.append(qst.perform_quantum_state_tomography(l, operator_circuit, nqubit, heralded_modes, post_process,
                                                        renormalization, brightness, g2, indistinguishability, loss))
    basis = matrix_basis(nqubit)
    L = np.zeros((d ** 2, d ** 2), dtype='complex_')
    for j in range(d ** 2):
        rhoj = rho(j, nqubit)
        mu = decomp(rhoj, basis)
        eps_rhoj = sum([mu[i] * EPS[i] for i in range(d ** 2)])
        for k in range(d ** 2):
            L[j, k] = eps_rhoj[k // d, k % d]
    return matrix_to_vector(L)


def chi_mult_qubit(operator_circuit, nqubit, heralded_modes=[], post_process=False, renormalization=None, brightness=1,
                   g2=0, indistinguishability=1, loss=0):
    """
    Computes the chi matrix of the operator

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
    :return: 2**(2*nqubit)x2**(2*nqubit) array
    """
    Binv = np.linalg.pinv(beta_matrix_mult_qubit(nqubit))
    L = lambd_mult_qubit(operator_circuit, nqubit, heralded_modes, post_process, renormalization, brightness, g2,
                         indistinguishability, loss)
    X = np.dot(Binv, L)
    return vector_to_matrix(X)


def state_to_dens_matrix(state):
    return np.dot(state, np.conjugate(np.transpose(state)))


def compute_matrix(j):
    if j == 0:
        return np.eye((2), dtype='complex_')
    if j == 1:
        return np.array([[0, 1], [1, 0]], dtype='complex_')
    if j == 2:
        return (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype='complex_')
    if j == 3:
        return (1 / np.sqrt(2)) * np.array([[1, 1], [1j, -1j]], dtype='complex_')


def matrix_basis(nqubit):  # create a matrix basis from all the tensor products states
    # needed for rho_j and to compute epsilon rho_j
    d = 2 ** nqubit
    B = []
    for j in range(d ** 2):
        v = np.zeros((d, 1), dtype='complex_')
        v[0] = 1
        k = []
        for m in range(nqubit - 1, -1, -1):
            k.append(j // (4 ** m))
            j = j % (4 ** m)
        k.reverse()
        M = compute_matrix(k[0])
        for i in k[1:]:
            M = np.kron(compute_matrix(i), M)
        B.append(state_to_dens_matrix(np.dot(M, v)))
    return B


def matrix_to_vector(matrix):  # concatenate a matrix d*d into a vector d**2
    # simply flatten() -> rows after row
    n = len(matrix[0])
    x = np.zeros((n ** 2), dtype='complex_')
    for i in range(n ** 2):
        x[i] = matrix[i // n, i % n]
    return x


def vector_to_matrix(vector):  # expand a vector d**2 into a matrix d*d
    n = len(vector)
    d = int(np.sqrt(n))
    M = np.zeros((d, d), dtype='complex_')
    for i in range(d):
        for j in range(d):
            if len(vector.shape) == 2:
                M[i, j] = vector[i * d + j][0]
            elif len(vector.shape) == 1:
                M[i, j] = vector[i * d + j]
    return M


def decomp(matrix, basis):  # linear decomposition of any matrix upon a basis
    # decomposition used in rho_j creation - process tomography
    n = len(matrix[0])
    y = matrix_to_vector(matrix)
    L = []
    for m in basis:
        L.append(matrix_to_vector(m))
    A = np.zeros((n ** 2, n ** 2), dtype='complex_')
    for i in range(n ** 2):
        for j in range(n ** 2):
            A[i, j] = L[j][i]
    x = np.dot(np.linalg.inv(A), y)
    return x


def lambda_mult_ideal(operator, nqubit):
    # not simulating perceval circuit, but simply a mathematical result for ideal gate
    # to compute process fidelity
    d = 2 ** nqubit
    L = np.zeros((d ** 2, d ** 2), dtype='complex_')
    for j in range(d ** 2):
        rhoj = rho(j, nqubit)
        eps_rhoj = np.linalg.multi_dot([operator, rhoj, np.conjugate(np.transpose(operator))])
        for k in range(d ** 2):
            L[j, k] = eps_rhoj[k // d, k % d]
    L1 = np.zeros((d ** 4, 1), dtype='complex_')
    for i in range(d ** 4):
        L1[i] = L[i // (d ** 2), i % (d ** 2)]
    return L1


def chi_mult_ideal(operator, nqubit):
    X = np.dot(np.linalg.pinv(beta_matrix_mult_qubit(nqubit)), lambda_mult_ideal(operator, nqubit))
    return vector_to_matrix(X)


# ##### Fidelity calculations ######################################################################################
def process_fidelity(operator, operator_circuit, heralded_modes=[], post_process=False, renormalization=None,
                     brightness=1, g2=0, indistinguishability=1, loss=0):
    """
    Computes the process fidelity of an operator and its perceval circuit

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
    X0 = chi_mult_qubit(operator_circuit, nqubit, heralded_modes, post_process, renormalization, brightness, g2,
                        indistinguishability, loss)
    X1 = chi_mult_ideal(operator, nqubit)
    return np.real(np.trace(np.dot(X0, X1)))


def random_fidelity(nqubit, brightness=1, g2=0, indistinguishability=1, loss=0):
    """
    Computes the process and the average fidelity of a random non-entangling operator

    :param nqubit: number of qubits
    :param brightness source brightness
    :param g2 SPS g2
    :param indistinguishability photon indistinguishability
    :param loss known losses in source

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
    pf = process_fidelity(M, CU, brightness=brightness, g2=g2, indistinguishability=indistinguishability, loss=loss)
    afq = average_fidelity(M, CU, brightness=brightness, g2=g2, indistinguishability=indistinguishability, loss=loss)
    print('process fidelity :', pf, '\n',
          'average fidelity :', afq)


def map_reconstructed(rho, operator_circuit, nqubit, heralded_modes=[], post_process=False, renormalization=None,
                      brightness=1, g2=0, indistinguishability=1, loss=0):
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
    X = chi_mult_qubit(operator_circuit, nqubit, heralded_modes, post_process, renormalization, brightness, g2,
                       indistinguishability, loss)
    eps = np.zeros((d, d), dtype='complex_')
    for m in range(d ** 2):
        for n in range(d ** 2):
            eps += X[m, n] * np.linalg.multi_dot([E(m, nqubit), rho, np.transpose(np.conjugate(E(n, nqubit)))])
    # Eqn 2.4 the exact sum
    return eps


def average_fidelity_with_reconstruction(operator, operator_circuit, heralded_modes=[], post_process=False,
                                         renormalization=None, brightness=1, g2=0, indistinguishability=1, loss=0):
    """
    not so important-computes avg fideltiy in a longer way
    Computes the average fidelity of an operator and its perceval circuit by recontruction of the whole map

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
    for j in range(d ** 2):
        Uj = E(j, nqubit)
        Ujdag = np.transpose(np.conjugate(Uj))
        eps_Uj = map_reconstructed(Uj, operator_circuit, nqubit, heralded_modes, post_process, renormalization,
                                   brightness, g2, indistinguishability, loss)
        a = np.linalg.multi_dot([operator, Ujdag, Udag, eps_Uj])
        f += (1 / ((d + 1) * (d ** 2))) * np.trace(a)
    return np.real(f)


def error_process_matrix(operator, operator_circuit, heralded_modes=[], post_process=False, renormalization=None,
                         brightness=1, g2=0, indistinguishability=1, loss=0):
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
    X = chi_mult_qubit(operator_circuit, nqubit, heralded_modes, post_process, renormalization, brightness, g2,
                       indistinguishability, loss)
    V = np.zeros((d ** 2, d ** 2), dtype='complex_')
    for m in range(d ** 2):
        for n in range(d ** 2):
            Emdag = np.transpose(np.conjugate(E(m, nqubit)))
            En = E(n, nqubit)
            Udag = np.transpose(np.conjugate(operator))
            V[m, n] = (1 / d) * np.trace(np.linalg.multi_dot([Emdag, En, Udag]))
    return np.linalg.multi_dot([V, X, np.conjugate(np.transpose(V))])


def average_fidelity(operator, operator_circuit, heralded_modes=[], post_process=False, renormalization=None,
                     brightness=1, g2=0, indistinguishability=1, loss=0):
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
                                                        renormalization, brightness, g2, indistinguishability, loss))

    basis = matrix_basis(nqubit)
    for j in range(d ** 2):
        Uj = E(j, nqubit)
        mu = decomp(Uj, basis)
        eps_Uj = sum([mu[i] * EPS[i] for i in range(d ** 2)])  # compute the map on a basis
        Ujdag = np.transpose(np.conjugate(Uj))
        a = np.linalg.multi_dot([operator, Ujdag, Udag, eps_Uj])
        f += (1 / ((d + 1) * (d ** 2))) * np.trace(a)
    return np.real(f)


def mixture(operator_circuit, nqubit, heralded_modes=[], post_process=False, renormalization=None, brightness=1, g2=0,
            indistinguishability=1, loss=0):
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
    X = chi_mult_qubit(operator_circuit, nqubit, heralded_modes, post_process, renormalization, brightness, g2,
                       indistinguishability, loss)
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
