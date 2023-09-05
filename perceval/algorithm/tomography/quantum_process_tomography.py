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
import itertools
from scipy.stats import unitary_group
from perceval.simulators import Simulator
from perceval.backends import SLOSBackend

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

# ##### gates ######################################################################################
# def cz_heralded():
#     # perceval circuit (w/o postselection or herald) for 6 modes heralded CZ
#     theta1 = 2 * np.pi * 54.74 / 180
#     theta2 = 2 * np.pi * 17.63 / 180
#     last_modes_cz = (pcvl.Circuit(4)
#                      .add(0, PS(np.pi))
#                      .add(3, PS(np.pi))
#                      .add((1, 2), PERM([1, 0]))
#                      .add((0, 1), BS.H(theta=theta1))
#                      .add((2, 3), BS.H(theta=theta1))
#                      .add((1, 2), PERM([1, 0]))
#                      .add((0, 1), BS.H(theta=-theta1))
#                      .add((2, 3), BS.H(theta=theta2)))
#     c_hcz = (pcvl.Circuit(6, name="Heralded CZ")
#              .add((1, 2), PERM([1, 0]))
#              .add((2, 3, 4, 5), last_modes_cz, merge=True)
#              .add((1, 2), PERM([1, 0])))
#     return c_hcz
#
#
# def cnot_from_cz():
#     # building a CNOT from the heralded CZ using CX=(IxH)CZ(IxH)
#     cnot = pcvl.Circuit(6, name="Heralded CNOT")
#     cnot.add(2, BS.H())
#     cnot.add(0, cz_heralded())
#     cnot.add(2, BS.H())
#     return cnot


# def heralded_cnot():
#     # perceval circuit (w/o postselection or herald) for 8 modes heralded CNOT
#     R1 = 0.228
#     R2 = 0.758
#     theta1 = BS.r_to_theta(R1)
#     theta2 = BS.r_to_theta(R2)
#     c_hcnot = (pcvl.Circuit(8, name="Heralded CNOT")
#                .add(1, PERM([2, 4, 3, 0, 1]))
#                .add(4, BS.H())
#                .add(3, PERM([1, 3, 0, 4, 2]))
#                .add(3, BS.H())
#                .add(3, PERM([2, 0, 1]))
#                .add(2, BS.H(theta=theta1))
#                .add(4, BS.H(theta=theta1))
#                .add(3, PERM([1, 2, 0]))
#                .add(3, BS.H())
#                .add(1, PERM([2, 0, 3, 1, 6, 5, 4]))
#                .add(2, BS.H(theta=theta2))
#                .add(2, PERM([1, 0]))
#                .add(4, BS.H(theta=theta2))
#                .add(4, PERM([1, 2, 0]))
#                .add(4, BS.H())
#                .add(1, PERM([4, 3, 0, 2, 1])))
#     return c_hcnot
#
#
# def post_processed_cnot():
#     # perceval circuit (w/o postselection or herald) for 6 modes postprocessed CNOT
#     theta_13 = BS.r_to_theta(1 / 3)
#     c_cnot = (pcvl.Circuit(6, name="PostProcessed CNOT")
#               .add(0, PERM([0, 2, 3, 4, 1]))
#               .add((0, 1), BS.H(theta_13))
#               .add((0, 1), PERM([1, 0]))
#               .add((3, 4), BS.H())
#               .add((2, 3), PERM([1, 0]))
#               .add((2, 3), BS.H(theta_13))
#               .add((2, 3), PERM([1, 0]))
#               .add((4, 5), BS.H(theta_13))
#               .add((3, 4), BS.H())
#               .add(0, PERM([4, 0, 1, 2, 3])))
#     return c_cnot
#
# ##### Important matrices Class Process tomography ######################################################################################
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
def prep_circ_single_qubit(j):
    """
    Prepares one photon in each of the following states: |1,0>,|0,1>,|+>,|+i>

    :param j: int between 0 and 3
    :return: 2 modes perceval circuit
    """
    prep_circuit = pcvl.Circuit(2)
    if j == 1:
        prep_circuit.add(0, PERM([1, 0]))
    if j // 2 == 1:
        prep_circuit.add(0, BS.H())
    if j == 3:
        prep_circuit.add(1, PS(np.pi / 2))
    return prep_circuit  # todo: ask how to choose which circuit


def prep_circuit_multi_qubit(j, nqubit):
    """
    Prepares each photon in each of the following states: |1,0>,|0,1>,|+>,|+i>

    :param j: int between 0 and 4**nqubit-1
    :param nqubit: number of qubits
    :return: 2*nqubit modes perceval circuit
    """
    prep_circuit = pcvl.Circuit(2 * nqubit)
    for m in range(len(j)):
        prep_circuit.add(2 * m, prep_circ_single_qubit(j[m]), merge=True)
    return prep_circuit


# ##### measurement circuit ######################################################################################
def meas_circ_single_qubit(j):
    """
    Measures the photon in the pauli basis I,X,Y,Z

    :param j: int between 0 and 3
    :return: 2 modes perceval circuit
    """
    meas_circuit = pcvl.Circuit(2)
    if j == 1:
        meas_circuit.add(0, BS.H())
    if j == 2:
        meas_circuit.add(0, BS.Rx(theta=np.pi / 2, phi_bl=np.pi, phi_br=-np.pi / 2))
    return meas_circuit


def measurement_circuit(j, nqubit):
    """
     Measures each photon in the pauli basis

    :param j: int between 0 and 4**nqubit-1
    :param nqubit: number of qubits
    :return: 2*nqubit modes perceval circuit
    """
    meas_circuit = pcvl.Circuit(2 * nqubit)
    for m in range(len(j)):
        meas_circuit.add(2 * m, meas_circ_single_qubit(j[m]), merge=True)
    return meas_circuit


# ##### P and Stokes are part of QST ##############################################################################
def P(k, n):
    # set of subsets of size k in {0,...,n-1}
    s = {i for i in range(n)}
    return list(itertools.combinations(s, k))


def stokes_parameter(num_state, operator_circuit, i, heralded_modes=[], post_process=False, renormalization=None,
                     brightness=1, g2=0, indistinguishability=1, loss=0):
    """
    Computes the Stokes parameter S_i for state num_state after operator_circuit

    :param num_state: list of length of number of qubits representing the preparation circuit
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
    qpt_circuit = pcvl.Circuit(2 * nqubit + len(heralded_modes))
    # state preparation
    qpt_circuit.add(0, prep_circuit_multi_qubit(num_state, nqubit))
    # unknown operator
    qpt_circuit.add(0, operator_circuit)
    # measurement operator
    qpt_circuit.add(0, measurement_circuit(i, nqubit))

    source = pcvl.Source(emission_probability=brightness, multiphoton_component=g2,
                         indistinguishability=indistinguishability, losses=loss)
    simulator = Simulator(SLOSBackend())
    simulator.set_circuit(qpt_circuit)

    # postselection if no renormalization
    if renormalization is None:
        ps = pcvl.PostSelect()
        if post_process:
            for m in range(nqubit):
                ps.eq([2 * m, 2 * m + 1], 1)
        for m in heralded_modes:
            ps.eq([m[0]], m[1])
        simulator.set_postselection(ps)

    # input state accounting the heralded modes
    input_state = pcvl.BasicState("|1,0>")
    for _ in range(1, nqubit):
        input_state *= pcvl.BasicState("|1,0>")
    for m in heralded_modes:
        input_state *= pcvl.BasicState([m[1]])
    input_distribution = source.generate_distribution(expected_input=input_state)

    simulator.set_min_detected_photon_filter(0)
    output_distribution = simulator.probs_svd(input_distribution)["results"]

    s = 0  # calculation of the Stokes parameter
    for k in range(nqubit + 1):
        for J in P(k, nqubit):
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
            s += eta * output_distribution[measurement_state]
    if renormalization is None:
        return s
    return s / renormalization


def quantum_state_tomography_mult(state, operator_circuit, nqubit, heralded_modes=[], post_process=False,
                                  renormalization=None, brightness=1, g2=0, indistinguishability=1, loss=0):
    """
    Suffix mult not necesary- works for both single and multi qubit.

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
        density_matrix += stokes_parameter(state, operator_circuit, i, heralded_modes, post_process, renormalization,
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
        EPS.append(
            quantum_state_tomography_mult(l, operator_circuit, nqubit, heralded_modes, post_process, renormalization,
                                          brightness, g2, indistinguishability, loss))
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
    #Eqn 2.4 the exact sum
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
        EPS.append(
            quantum_state_tomography_mult(l, operator_circuit, nqubit, heralded_modes, post_process, renormalization,
                                          brightness, g2, indistinguishability, loss))

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

# FIDELITY OPTIMIZER MAKES IMPROVEMENTS. MAYBE NEEDED, NOT WORKING
###################
# from here onward -> connected to some formula to calculate fidelity in Ascella paper. Stephen and Rawad
def int_to_vector(i, nqubit):
    if nqubit == 1:
        if i == 0:
            return np.array([[1], [0]], dtype='complex_')
        return np.array([[0], [1]], dtype='complex_')
    vec = int_to_vector(i // (2 ** (nqubit - 1)), 1)
    i = i % (2 ** (nqubit - 1))
    for k in range(nqubit - 2, -1, -1):
        vec = np.kron(vec, int_to_vector(i // (2 ** k), 1))
        i = i % (2 ** k)
    return vec


def alpha(i, j, k, l, U):
    nqubit = int(np.log2(len(U)))
    vec_i, vec_j, vec_k, vec_l = int_to_vector(i, nqubit), int_to_vector(j, nqubit), int_to_vector(k, nqubit), \
        int_to_vector(l, nqubit)
    Udag = np.transpose(np.conjugate(U))
    idag, ldag = np.transpose(np.conjugate(vec_i)), np.transpose(np.conjugate(vec_l))
    x = np.linalg.multi_dot([idag, Udag, vec_k])[0, 0]
    y = np.linalg.multi_dot([ldag, U, vec_j])[0, 0]
    return x * y


def int_to_list(i, nqubit):
    l = []
    for k in range(nqubit - 1, -1, -1):
        l.append(i // (4 ** k))
        i = i % (4 ** k)
    return l


"""def fast_average_fidelity_2(operator,operator_circuit,heralded_modes=[],post_process=False,brightness=1,g2=0,
indistinguishability=1,loss=0):
    nqubit=int(np.log2(len(operator)))
    d=2**nqubit
    C=1/(d*(d+1))
    basis=matrix_basis(nqubit)
    M=np.zeros((d**2,d**2),dtype='complex_')
    for i0 in range(d):
        for j0 in range(d):
            for i1 in range(d):
                for j1 in range(d):
                    a=alpha(i1,j1,i0,j0,operator)+alpha(i1,j1,j0,i0,operator)
                    if a!=0:
                        meas=np.dot(int_to_vector(i0,nqubit),np.conjugate(np.transpose(int_to_vector(j0,nqubit))))
                        state=np.dot(int_to_vector(i1,nqubit),np.conjugate(np.transpose(int_to_vector(j1,nqubit))))
                        mu,nu=decomp(meas,basis),decomp(state,basis)
                        for l in range(d**2):
                            for k in range(d**2):
                                M[l,k]+=a*mu[k]*nu[l]
    t=0
    for x in range(d**2):
        y0,z0=M[x,0],M[0,x]
        ybool,zbool=True,True
        for y in range(1,d):
            if M[x,y]!=y0:
                ybool=False
            if M[y,x]!=z0:
                zbool=False
        if ybool:
            return False

    return C*M

def fast_average_fidelity(operator,operator_circuit,heralded_modes=[],post_process=False,brightness=1,g2=0,indistinguishability=1,loss=0):
    nqubit=int(np.log2(len(operator)))
    d=2**nqubit
    C=1/(d*(d+1))
    s=0
    basis=matrix_basis(nqubit)
    memory={}
    for i0 in range(d):
        for j0 in range(d):
            for i1 in range(d):
                for j1 in range(d):
                    a=alpha(i1,j1,i0,j0,operator)+alpha(i1,j1,j0,i0,operator)
                    if a!=0:
                        print(a)
                        meas=np.dot(int_to_vector(i0,nqubit),np.conjugate(np.transpose(int_to_vector(j0,nqubit))))
                        state=np.dot(int_to_vector(i1,nqubit),np.conjugate(np.transpose(int_to_vector(j1,nqubit))))
                        mu,nu=decomp(meas,basis),decomp(state,basis)
                        #print(mu,nu)
                        t=0
                        id_present=True #check if you can use CPTP property
                        x=nu[0]
                        for l0 in range(1,d):
                            if nu[l0]!=x:
                                id_present=False
                        start=0
                        if id_present:
                            start=d
                            t+=nu[l]*np.trace(meas)
                        for l in range(start,d**2):
                            if nu[l]!=0:
                                for k in range(d**2):
                                    if mu[k]!=0:
                                        print(l,k)
                                        if (l,k) not in memory.keys():
                                            state1=int_to_list(l,nqubit)
                                            meas1=int_to_list(k,nqubit)
                                            num_meas=[]
                                            for u in meas1:
                                                if u==0 or u==1:
                                                    num_meas.append((0,int((-1)**u)))
                                                else:
                                                    num_meas.append((u-1,1))
                                            num_meas=[(u,1) for u in meas1]
                                            memory[(l,k)]=qpt_circuit(state1,operator_circuit,num_meas,nqubit,heralded_modes=[],post_process=False,brightness=1,g2=0,indistinguishability=1,loss=0)
                                        t+=mu[k]*nu[l]*memory[(l,k)]
                    s+=a*t
    return np.real(C*s),memory"""
