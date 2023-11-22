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

import os
from pathlib import Path

import pytest
import numpy as np
from scipy.stats import unitary_group

import perceval as pcvl
from perceval.components import catalog, Processor, Circuit, PauliType
from perceval.backends import SLOSBackend
from perceval.components import Unitary
from perceval.algorithm import ProcessTomography, StateTomography
from perceval.algorithm.tomography.tomography_utils import is_physical, get_preparation_circuit, \
    _generate_pauli_index, _vector_to_sq_matrix, _matrix_to_vector, _matrix_basis, _coef_linear_decomp

from _test_utils import _check_image

CNOT_TARGET = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype='complex_')
TEST_IMG_DIR = Path(__file__).resolve().parent / 'imgs'


@pytest.mark.parametrize("pauli_gate", [PauliType.I, PauliType.X, PauliType.Y, PauliType.Z])
def test_density_matrix_state_tomography(pauli_gate):
    p = Processor("Naive", Circuit(2) // get_preparation_circuit(pauli_gate))  # 1 qubit Pauli X gate
    qst = StateTomography(operator_processor=p)
    density_matrix = qst.perform_state_tomography([PauliType.X])

    res = is_physical(density_matrix, nqubit=1)

    assert len(density_matrix) == 2
    assert res['Trace=1'] is True
    assert res["Hermitian"] is True
    assert res["Completely Positive"] is True


def fidelity_op_process_tomography(op, op_proc):
    # create process tomography object
    qpt = ProcessTomography(operator_processor=op_proc)
    # compute Chi matrix
    chi_op_ideal = qpt.chi_target(op)
    chi_op = qpt.chi_matrix()
    # Compute fidelity
    op_fidelity = qpt.process_fidelity(chi_op, chi_op_ideal)
    return op_fidelity


def test_fidelity_klm_cnot():
    # set operator circuit, num qubits
    cnot_p = catalog["klm cnot"].build_processor()
    cnot_fidelity = fidelity_op_process_tomography(CNOT_TARGET, cnot_p)
    assert cnot_fidelity == pytest.approx(1)


def test_fidelity_postprocessed_cnot():
    # set operator circuit, num qubits
    cnot_p = catalog["postprocessed cnot"].build_processor()
    cnot_fidelity = fidelity_op_process_tomography(CNOT_TARGET, cnot_p)
    assert cnot_fidelity == pytest.approx(1)


def test_fidelity_random_op():
    # process tomography to compute fidelity of a random 2 qubit gate operation
    nqubit = 2
    L = []
    for i in range(nqubit):
        L.append(unitary_group.rvs(2))

    random_op = L[0]
    random_op_circ = pcvl.Circuit(2 * nqubit).add(0, Unitary(pcvl.Matrix(L[0])))

    for i in range(1, nqubit):
        random_op = np.kron(random_op, L[i])
        random_op_circ.add(2 * i, Unitary(pcvl.Matrix(L[i])))

    random_op_proc = Processor(backend=SLOSBackend(), m_circuit=random_op_circ.m)
    random_op_proc.add(0, random_op_circ)

    random_op_fidelity = fidelity_op_process_tomography(random_op, random_op_proc)

    assert random_op_fidelity == pytest.approx(1)


def test_chi_cnot_is_physical_and_display():
    cnot_p = catalog["klm cnot"].build_processor()

    qpt = ProcessTomography(operator_processor=cnot_p)

    chi_op = qpt.chi_matrix()
    res = is_physical(chi_op, nqubit=2)

    assert res['Trace=1'] is True  # if Chi has Trace = 1
    assert res['Hermitian'] is True  # if Chi is Hermitian
    assert res['Completely Positive'] is True  # if input Chi is Completely Positive

    # display
    curr_path = "tomography_cnot.svg"
    other_path = TEST_IMG_DIR / Path("tomography_cnot.svg")
    pcvl.pdisplay_to_file(qpt, path=curr_path)
    is_same = _check_image(curr_path, other_path)
    os.remove(curr_path)
    assert is_same

def test_processor_odd_modes():
    # tests that a generic processor with odd number of modes does not work
    with pytest.raises(ValueError):
        ProcessTomography(operator_processor=Processor(SLOSBackend(), m_circuit=5))


def test_generate_pauli():
    pauli_idx = _generate_pauli_index(2)
    for elem in pauli_idx:
        assert all(isinstance(val, PauliType) for val in elem)


def test_utils_vector_to_sq_matrix():
    # tests a valid vector that can be reshaped into a square matrix
    m = _vector_to_sq_matrix(np.ones(16))
    assert m.shape == (4, 4)
    # tests an incorrect input to reshape to square matrix
    with pytest.raises(ValueError):
        _vector_to_sq_matrix(np.ones(10))


def test_matrix_to_vector():
    v = _matrix_to_vector(np.eye(5))
    assert len(v) == 25


def test_matrix_basis_n_decomp():
    # generate matrix basis for the case with nqubit=1
    basis = _matrix_basis(nqubit=1, d=2)

    # any random matrix of the corresponding size
    matrix = pcvl.MatrixN(np.random.randint(1, 10, (2, 2)))

    # decomposing matrix into basis
    mu = _coef_linear_decomp(matrix, basis)

    # reconstruct
    matrix_rebuilt = pcvl.MatrixN(np.zeros(matrix.shape, dtype=complex))

    for idx, basis_matrices in enumerate(basis):
        matrix_rebuilt += mu[idx]*basis_matrices

    assert np.allclose(matrix, matrix_rebuilt)
