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

from perceval.components import get_preparation_circuit, get_measurement_circuit
from typing import List


class StatePreparation:
    """
    Builds preparation circuits to prepare an input photon in each of the following
    logical qubit state states: |0>,|1>,|+>,|+i> using Pauli Gates.

    :param prep_state_indices: List of 'n'(=nqubit) indices to choose one of the logical states for each qubit
    """
    def __init__(self, prep_state_indices: List):
        self._prep_state_indices = prep_state_indices

    def __iter__(self):
        """
        Returns preparation circuits qubit by qubit
        """
        for i, pauli_type in enumerate(self._prep_state_indices):
            yield i*2, get_preparation_circuit(pauli_type)


class MeasurementCircuit:
    """
    Builds a measurement circuit to measure photons created in the Pauli Basis (I,X,Y,Z) to perform
    tomography experiments.

    :param pauli_indices: List of 'n'(=nqubit) indices to choose a circuit to measure the prepared state at nth qubit
    """

    def __init__(self, pauli_indices: List):
        self._pauli_indices = pauli_indices

    def __iter__(self):
        for i, pauli_type in enumerate(self._pauli_indices):
            yield i*2, get_measurement_circuit(pauli_type)
