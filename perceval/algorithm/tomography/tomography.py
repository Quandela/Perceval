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

from perceval.components import Circuit, Processor, BS, Port, get_pauli_circuit
from perceval.algorithm.abstract_algorithm import AAlgorithm
from perceval.utils import BasicState, Encoding
from perceval.utils.postselect import PostSelect
from typing import List
from .tomography_utils import _matrix_basis, _matrix_to_vector, _vector_to_sq_matrix, _coef_linear_decomp, \
    _get_fixed_basis_ops, _get_canonical_basis_ops, _krauss_repr_ops, _generate_pauli_index, _list_subset_k_from_n


class StatePreparationCircuit(Circuit):
    """
    Builds a preparation circuit to prepare an input photon in each of the following
    logical Qubit state states: |0>,|1>,|+>,|+i> using Pauli Gates.

    :param nqubit: Number of Qubits
    :param prep_state_indices: List of 'n'(=nqubit) indices to choose one of the logical states for each qubit
    """

    def __init__(self, prep_state_indices: List, nqubit: int):
        super().__init__(m=2*nqubit, name="State Preparation Circuit")
        self._nqubit = nqubit
        self._prep_state_indices = prep_state_indices

    def build_preparation_circuit(self):
        """
        Builds the preparation circuit for tomography experiment
        """
        for m in range(len(self._prep_state_indices)):
            self.add(2 * m, get_pauli_circuit(self._prep_state_indices[m]), merge=True)
        return self


class MeasurementCircuit(Circuit):
    """
    Builds a measurement circuit to measure photons created in the Pauli Basis (I,X,Y,Z) to perform
    tomography experiments.

    :param nqubit: Number of Qubits
    :param meas_pauli_basis_indices: List of 'n'(=nqubit) indices to choose a circuit to measure
    the prepared state at nth Qubit
    """

    def __init__(self, meas_pauli_basis_indices: List, nqubit: int):
        super().__init__(m=2*nqubit, name="Measurement Basis Circuit")
        self._nqubit = nqubit
        self._meas_pauli_basis_indices = meas_pauli_basis_indices

    @staticmethod
    def _meas_circ(pauli_meas_circ_index: int) -> Circuit:
        # Prepares 1 qubit circuits to measure a photon in the pauli basis I,X,Y,Z
        # param pauli_meas_circ_index: int between 0 and 3
        # returns a 2 modes Measurement Circuit

        assert 0 <= pauli_meas_circ_index <= 3, f'Invalid index for measurement circuit'

        if pauli_meas_circ_index == 1:
            return Circuit(2) // (0, BS.H())
        elif pauli_meas_circ_index == 2:
            return Circuit(2) // (0, BS.Rx(theta=np.pi / 2, phi_bl=np.pi, phi_br=-np.pi / 2))
        else:
            return Circuit(2)

    def build_measurement_circuit(self):
        """
        Builds the circuit to perform measurement of photons prepared in the Pauli basis
        """
        for m in range(len(self._meas_pauli_basis_indices)):
            self.add(2 * m, self._meas_circ(self._meas_pauli_basis_indices[m]), merge=True)
        return self


class StateTomography(AAlgorithm):
    """
    Experiment to reconstruct the state of the system by tomography experiment.
    - Adds preparation and measurement circuits to input processor (with the gate operation under study)
    - Computes parameters required to do state tomography
    - Performs Tomography experiment - Computes and Returns density matrices for each input state
    """
    def __init__(self, nqubit: int, operator_processor: Processor, post_process=False, renormalization=None):
        super().__init__(processor=operator_processor)
        self._nqubit = nqubit
        self._operator_processor = operator_processor  # Gate operation under study
        self._backend = operator_processor.backend  # default - SLOSBackend()
        self._source = operator_processor.source  # default - ideal source
        self._post_process = post_process
        self._renormalization = renormalization
        self._heralded_modes = [(key, value) for key, value in operator_processor.heralds.items()]
        self._size_hilbert = 2 ** nqubit
        # Todo: I need to put this number somewhere, i forgot. find [number of elements = between (0 -> 4**nqubits-1)]

    def _input_state_dist_config(self):
        # Configures the input state for the Processor
        input_state = BasicState("|1,0>")
        for _ in range(1, self._nqubit):
            # setting the input state for the gate qubit modes
            input_state *= BasicState("|1,0>")
        for m in self._heralded_modes:
            # setting the input for heralded modes of the given processor
            input_state *= BasicState([m[1]])
        input_distribution = self._source.generate_distribution(expected_input=input_state)
        return input_distribution

    def _compute_probs(self, prep_state_indices, meas_pauli_basis_indices):
        # Adds preparation and measurement circuit to input processor (with the gate operation under study)
        # and computes the output probability distribution.
        # param prep_state_indices: List of "nqubit" indices selecting the circuit at each qubit for a preparation state
        # param meas_pauli_basis_indices: List of "nqubit" indices selecting the circuit at each qubit for a measurement
        # circuit
        # return: Output state probability distribution

        pc = StatePreparationCircuit(prep_state_indices, self._nqubit)  # state preparation circuit object
        mc = MeasurementCircuit(meas_pauli_basis_indices, self._nqubit)  # measurement basis circuit object

        p = Processor(self._backend, self._nqubit*2, self._source)
        # A Processor with identical backend and source as the input
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

    def _stokes_parameter(self, prep_state_indices, meas_pauli_basis_indices):
        # Computes the Stokes parameter S_i for state prep_state_indices after operator_circuit
        # param prep_state_indices: list of length of number of qubits representing the preparation circuit
        # param meas_pauli_basis_indices: list of length of number of qubits representing the measurement circuit and the
        # eigenvector we are measuring
        # returns the value of Stokes parameter for a given combination of input and output state -> a complex float

        output_distribution = self._compute_probs(prep_state_indices, meas_pauli_basis_indices)

        # calculation of the Stokes parameter begins here
        stokes_param = 0
        for k in range(self._nqubit + 1):
            for J in _list_subset_k_from_n(k, self._nqubit):
                eta = 1
                if 0 not in J:
                    measurement_state = BasicState("|1,0>")
                else:
                    measurement_state = BasicState("|0,1>")
                    if meas_pauli_basis_indices[0] != 0:
                        eta *= -1
                for j in range(1, self._nqubit):
                    if j not in J:
                        measurement_state *= BasicState("|1,0>")
                    else:
                        measurement_state *= BasicState("|0,1>")
                        if meas_pauli_basis_indices[j] != 0:
                            eta *= -1
                for m in self._heralded_modes:
                    measurement_state *= BasicState([m[1]])
                stokes_param += eta * output_distribution[measurement_state]

        if self._renormalization is None:
            return stokes_param
        return stokes_param / self._renormalization

    def perform_state_tomography(self, prep_state_indices):
        """
        Computes the density matrix of a state after the operator_circuit. Size d x d where d=size_of_hilbert_space

        :param prep_state_indices: list of length of number of qubits to index the corresponding preparation circuit
        :return: density matrix for a given input state preparation. size_hilbert x size_hilbert array.
        """
        density_matrix = np.zeros((self._size_hilbert, self._size_hilbert), dtype='complex_')

        pauli_indices = _generate_pauli_index(self._nqubit)
        for index, elem in enumerate(pauli_indices):
            elem_values = [p.value for p in elem]  # converting object to list of values
            density_matrix += self._stokes_parameter(prep_state_indices, elem_values) \
                              * _get_fixed_basis_ops(index, self._nqubit)
        density_matrix = ((1 / 2) ** self._nqubit) * density_matrix
        return density_matrix


class ProcessTomography(AAlgorithm):
    """
    Experiment to reconstruct the process map of the gate operation by tomography experiment.
    - Computes the mathematical tensors/matrices defined by theory required to perform process tomography
    - Computes r$\chi$ matrix form of the operation process map
    - Provides analysis methods to investigate the results of process tomography
        -- Fidelity of the operation, Error process map

    """
    def __init__(self, nqubit: int, operator_processor: Processor, post_process=False,
                 renormalization=None):
        super().__init__(processor=operator_processor)
        self._nqubit = nqubit
        self._operator_processor = operator_processor
        self._backend = operator_processor.backend  # default - SLOSBackend()
        self._post_process = post_process
        self._renormalization = renormalization
        self._size_hilbert = 2 ** nqubit
        self._qst = StateTomography(nqubit=self._nqubit, operator_processor=self._operator_processor,
                                    post_process=self._post_process, renormalization=self._renormalization)

    def _beta_tensor_elem(self, j, k, m, n, nqubit):
        # computes the elements of beta^{mn}_{jk}, a rank 4 tensor, each index of which can
        # take values between 0 and d^2-1  [d = _size_hilbert]

        b = _krauss_repr_ops(m, _get_canonical_basis_ops(j, self._size_hilbert), n, nqubit)
        q, r = divmod(k, self._size_hilbert)  # quotient, remainder
        return b[q, r]

    def _beta_as_matrix(self):
        # compiles the 2D beta matrix by extracting elements of the rank 4 tensor computed by method _beta_tensor_elem

        num_meas = self._size_hilbert ** 4  # Total number of measurements needed for process tomography
        beta_matrix = np.zeros((num_meas, num_meas), dtype='complex_')
        for a in range(num_meas):
            j, k = divmod(a, self._size_hilbert ** 2)  # returns quotient, remainder
            for b in range(num_meas):
                # j,k,m,n are indices for _beta_tensor_elem
                # todo: fix tue morning - cool idea to remove divmod
                #
                # the task that all these are doing is creating pair of indices i,j xhich is a product of
                # a set with itself {0,1,2,...,n}x{0,1,2,...,n} = {(0,0),(0,1),(0,2),...,(0,n),...(n,n)}
                # only n changes but is mostly d**2
                #
                m, n = divmod(b, self._size_hilbert ** 2)
                beta_matrix[a, b] = self._beta_tensor_elem(j, k, m, n, self._nqubit)
        return beta_matrix

    def _lambda_vector(self):
        """
        Computes the lambda vector of the operator

        :return: size_hilbert**4 vector
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

    def _lambda_target(self, operator):
        # Implements a mathematical formula for ideal gate (given operator) to compute process fidelity
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

    def chi_matrix(self):
        """
        Computes the chi matrix of the operator_circuit. Size d^4 x d^4

        :return: 2**(2*nqubit)x2**(2*nqubit) array
        """
        beta_inv = np.linalg.pinv(self._beta_as_matrix())
        L = self._lambda_vector()
        X = np.dot(beta_inv, L)  # X is a vector here
        return _vector_to_sq_matrix(X)

    def chi_target(self, operator):
        # Implements a mathematical formula for ideal gate (given operator) to compute process fidelity
        beta_inv = np.linalg.pinv(self._beta_as_matrix())
        lambd = self._lambda_target(operator)
        X = np.dot(beta_inv, lambd)  # X is a matrix here
        return _vector_to_sq_matrix(X[:, 0])

    @staticmethod
    def process_fidelity(chi_computed, chi_ideal):
        """
        Computes the process fidelity of an operator (ideal) and its implementation (realistic)

        :param chi_computed: chi matrix computed from process tomography
        :param chi_ideal: Ideal chi matrix for the corresponding operator
        :return: float between 0 and 1
        """
        return np.real(np.trace(np.dot(chi_computed, chi_ideal)))

    def average_fidelity(self, operator):
        """
        Computes the average fidelity of an operator (ideal) and its implementation (realistic).
        This is not a full fidelity of the operation as given by the process_fidelity but
        simply that of the gate.

        :param operator: Gate matrix whose fidelity is to be calculated
        :return: float between 0 and 1
        """
        Udag = np.transpose(np.conjugate(operator))
        avg_fidelity = 1 / (self._size_hilbert + 1)

        # compute the map on a basis of states (tensor products of |0>, |1>, |+>,|i+>)
        density_matrices = []   # stores a list of density matrices for each measurement
        pauli_indices = _generate_pauli_index(self._nqubit)
        for prep_state_indices in pauli_indices:
            density_matrices.append(self._qst.perform_state_tomography(prep_state_indices))

        for j in range(self._size_hilbert ** 2):
            Uj = _get_fixed_basis_ops(j, self._nqubit)
            mu = _coef_linear_decomp(Uj, _matrix_basis(self._nqubit, self._size_hilbert))
            eps_Uj = sum([mu[i] * density_matrices[i] for i in range(self._size_hilbert ** 2)])
            # compute the map on a basis
            Ujdag = np.transpose(np.conjugate(Uj))
            a = np.linalg.multi_dot([operator, Ujdag, Udag, eps_Uj])
            avg_fidelity += (1 / ((self._size_hilbert + 1) * (self._size_hilbert ** 2))) * np.trace(a)
        return np.real(avg_fidelity)

    def error_process_matrix(self, computed_chi, operator):
        """
        Computes the error matrix for an operation from the computed chi
        Size d^4 x d^4

        :param computed_chi: chi matrix computed from process tomography
        :param operator: Gate (or operator) matrix
        :return: error process matrix of shape size_hilbert^4 x size_hilbert^4
        """
        V = np.zeros((self._size_hilbert ** 2, self._size_hilbert ** 2), dtype='complex_')
        for m in range(self._size_hilbert ** 2):
            for n in range(self._size_hilbert ** 2):
                Emdag = np.transpose(np.conjugate(_get_fixed_basis_ops(m, self._nqubit)))
                En = _get_fixed_basis_ops(n, self._nqubit)
                Udag = np.transpose(np.conjugate(operator))
                V[m, n] = (1 / self._size_hilbert) * np.trace(np.linalg.multi_dot([Emdag, En, Udag]))
        return np.linalg.multi_dot([V, computed_chi, np.conjugate(np.transpose(V))])
