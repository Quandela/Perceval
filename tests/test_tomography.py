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
from scipy.stats import unitary_group

import perceval
import perceval as pcvl
from perceval.components import catalog, Processor
from perceval.backends import SLOSBackend
from perceval.components import Unitary
from perceval.algorithm.tomography.quantum_process_tomography import QuantumProcessTomography


def fidelity_op_process_tomography(op, op_proc, nqubit):
    # create process tomography object
    qpt = QuantumProcessTomography(nqubit=nqubit, operator_processor=op_proc)
    # compute Chi matrix
    chi_op_ideal = qpt.chi_target(op)
    chi_op = qpt.chi_matrix()
    # Compute fidelity
    op_fidelity = qpt.process_fidelity(chi_op, chi_op_ideal)
    return op_fidelity


def test_fidelity_cnot_operator():
    # set operator circuit, num qubits
    cnot_p = catalog["klm cnot"].build_processor()
    cnot_op = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype='complex_')

    cnot_fidelity = fidelity_op_process_tomography(cnot_op, cnot_p, nqubit=2)

    assert cnot_fidelity == pytest.approx(1, 1e-3)  # computed fidelity is around 0.99967


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
    print(random_op_proc, random_op_proc.m)

    # random_U = perceval.Matrix.random_unitary(4)
    # random_op_proc = Processor(backend=SLOSBackend, m_circuit=4)
    # random_op_proc.add(0, random_U)

    pcvl.pdisplay(random_op_proc)
    random_op_fidelity = fidelity_op_process_tomography(random_op, random_op_proc, 2)

    assert random_op_fidelity == pytest.approx(1, 1e-6)


@pytest.mark.parametrize(("renorm", "expected"),
                         [(None, "|not Completely Positive|smallest eigenvalue :-0.00214"),
                          (0.0515, "|Completely Positive|")])
def test_chi_cnot_is_physical(renorm, expected):
    cnot_p = catalog["klm cnot"].build_processor()

    qpt = QuantumProcessTomography(nqubit=2, operator_processor=cnot_p, renormalization=renorm)

    chi_op = qpt.chi_matrix()
    res_is_physical = qpt.is_physical(chi_op)

    assert res_is_physical[0] == "|trace 1|"
    assert res_is_physical[1] == "|hermitian|"
    assert res_is_physical[2] == expected
