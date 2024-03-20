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

import pytest
import numpy as np
from perceval.components import (PauliType, PauliEigenStateType, processor_circuit_configurator, catalog, Circuit,
                                 Processor, get_pauli_eigen_state_prep_circ, get_pauli_gate)


def test_pauli_type():
    pauli_s = list(PauliType)
    pauli_val = [member.value for member in PauliType]
    is_ascending = np.all(np.diff(pauli_val) >= 0)
    assert len(pauli_s) == 4
    assert is_ascending == True


@pytest.mark.parametrize("pauli_eigen_states", [item for item in PauliEigenStateType])
def test_pauli_state_prep_circuits(pauli_eigen_states):
    c = Circuit(2) // get_pauli_eigen_state_prep_circ(pauli_eigen_states)
    assert c.m == 2


@pytest.mark.parametrize("pauli_gate", [PauliType.X, PauliType.Y, PauliType.Z])
def test_pauli_gates(pauli_gate):
    gate_matrix = get_pauli_gate(pauli_gate)
    assert np.trace(gate_matrix) == 0  # verify pauli matrices are traceless


def test_processor_circuit_configurator():

    with pytest.raises(TypeError):
        processor_circuit_configurator(Circuit(2),
                                       [PauliEigenStateType.Zm, PauliEigenStateType.Zm],
                                       [PauliType.I, PauliType.I],)

    cnot = catalog["klm cnot"].build_processor()
    with pytest.raises(TypeError):
        processor_circuit_configurator(cnot, [1, 0],[1, 0])

    configured_cnot = processor_circuit_configurator(cnot,
                                                     [PauliEigenStateType.Zm, PauliEigenStateType.Zm],
                                                     [PauliType.I, PauliType.I],)

    assert isinstance(configured_cnot, Processor)
