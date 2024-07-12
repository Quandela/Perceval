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

from typing import List
from .processor import AProcessor
from ._pauli import PauliType, PauliEigenStateType, get_pauli_eigen_state_prep_circ, get_pauli_basis_measurement_circuit


def _prep_state_circuit_preparer(prep_state_indices: List):
    """
    Generates a layer of state preparation circuits (essentially 1-qubit pauli gates) for each qubit.
    The logical qubit state prepared will be one of the list: |0>,|1>,|+>,|+i> using Pauli Gates.
    :param prep_state_indices: List of 'n'(=nqubit) indices to choose one of the logical states for each qubit
    """
    for i, pauli_type in enumerate(prep_state_indices):
        yield i * 2, get_pauli_eigen_state_prep_circ(pauli_type)


def _meas_state_circuit_preparer(pauli_indices: List):
    """
    Generates a layer of state measurement circuits (essentially measuring eigenstates of one of the pauli gates)
     for each qubit.
    :param pauli_indices: List of 'n'(=nqubit) indices to choose a circuit to measure the prepared state at nth qubit
    """
    for i, pauli_type in enumerate(pauli_indices):
        yield i*2, get_pauli_basis_measurement_circuit(pauli_type)


def processor_circuit_configurator(processor, prep_state_indices: list, meas_pauli_basis_indices: list):
    """
    Adds preparation and measurement circuit to input processor (with the gate operation under study) to configure
    it for the tomography experiment
    :param processor: Processor with input circuit on which Tomography is to be performed
    :param prep_state_indices: List of "nqubit" indices selecting the circuit at each qubit for a preparation state
    :param meas_pauli_basis_indices: List of "nqubit" indices selecting the circuit at each qubit for a measurement
     circuit
    :return: the configured processor to perform state tomography experiment
    """
    if not isinstance(processor, AProcessor):
        raise TypeError(f"{processor} is not a Processor and hence cannot be configured")

    if not all(isinstance(p_index, PauliEigenStateType) for p_index in prep_state_indices):
        raise TypeError(
            f"Indices for the preparation circuits should be a PauliEigenStateType")

    if not all(isinstance(m_index, PauliType) for m_index in meas_pauli_basis_indices):
        raise TypeError(
            f"Indices for the measurement circuits should be a PauliType")

    p = processor.copy()
    p.clear_input_and_circuit(processor.m)  # Clear processor content but keep its size

    for c in _prep_state_circuit_preparer(prep_state_indices):
        p.add(*c)  # Add state preparation circuit to the left of the operator

    p.add(0, processor)  # including the operator (as a processor)

    for c in _meas_state_circuit_preparer(meas_pauli_basis_indices):
        p.add(*c)  # Add measurement basis circuit to the right of the operator

    p.min_detected_photons_filter(0)  # QPU would have a problem with this

    return p
