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
import matplotlib.pyplot as plt
from scipy.stats import unitary_group
import perceval as pcvl
from perceval.components import catalog
from perceval.components import Unitary
from perceval.algorithm.tomography.quantum_process_tomography import QuantumStateTomography, QuantumProcessTomography, \
    FidelityTomography


def fidelity_op_process_tomography(op, op_circ, nqubit, herald):
    # create QST object
    qst = QuantumStateTomography(operator_circuit=op_circ, nqubit=nqubit, heralded_modes=herald)
    # create process tomography object and passing qst object as a parameter
    qpt = QuantumProcessTomography(nqubit=nqubit, operator_circuit=op_circ, qst=qst, heralded_modes=herald)
    #
    # compute Chi matrix
    chi_op_ideal = qpt.chi_target(op)
    chi_op = qpt.chi_matrix()
    # Compute fidelity
    op_fidelity = FidelityTomography(nqubit).process_fidelity(chi_op, chi_op_ideal)
    return op_fidelity


def test_fidelity_cnot_operator():
    # set operator circuit, num qubits
    cnot_circ = catalog["klm cnot"].build_circuit()
    cnot_op = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype='complex_')

    cnot_fidelity = fidelity_op_process_tomography(cnot_op, cnot_circ, nqubit=2, herald=[(4, 0), (5, 1), (6, 0), (7, 1)])

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

    random_op_fidelity = fidelity_op_process_tomography(random_op, random_op_circ, 2, herald=[])

    assert random_op_fidelity == pytest.approx(1, 1e-6)


def test_display_tomography():
    # set operator circuit, num qubits
    # Todo: move to pdisplay of perceval, fix input basis, make it generic
    cnot_circ = catalog["klm cnot"].build_circuit()
    cnot_op = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype='complex_')

    # create QST object
    qst = QuantumStateTomography(operator_circuit=cnot_circ, nqubit=2, heralded_modes=[(4, 0), (5, 1), (6, 0), (7, 1)])
    # create process tomography object and passing qst object as a parameter
    qpt = QuantumProcessTomography(nqubit=2, operator_circuit=cnot_op, qst=qst,
                                   heralded_modes=[(4, 0), (5, 1), (6, 0), (7, 1)])
    #
    # compute Chi matrix
    chi_op = qpt.chi_matrix()

    xx = np.linspace(0, len(chi_op), len(chi_op))
    yy = np.linspace(0, len(chi_op), len(chi_op))
    x, y = np.meshgrid(xx, yy)
    z = np.zeros(len(chi_op) * len(chi_op))  # z coordinates of each bar
    dx = np.ones(len(chi_op) * len(chi_op)) * 0.75  # Width of each bar
    dy = np.ones(len(chi_op) * len(chi_op)) * 0.75  # Depth of each bar

    fig = plt.figure(figsize=(6, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.bar3d(x.flatten(), y.flatten(), z, dx, dy, chi_op.real.flatten())
    ax.set_xlabel("basis")
    ax.set_ylabel("basis")
    ax.set_zlabel("chi real")
    ax.set_box_aspect(aspect=None, zoom=0.8)
    plt.show()

    fig = plt.figure(figsize=(6, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.bar3d(x.flatten(), y.flatten(), z, dx, dy, chi_op.imag.flatten())
    ax.set_xlabel("basis")
    ax.set_ylabel("basis")
    ax.set_zlabel("chi imag")
    ax.set_box_aspect(aspect=None, zoom=0.8)
    plt.show()
