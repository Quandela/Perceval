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
import perceval as pcvl
from perceval.components import catalog
from perceval.algorithm.tomography.quantum_process_tomography import QuantumStateTomography, QuantumProcessTomography, \
    FidelityTomography


# operator
def operotor_to_test():
    return np.array([[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1],
                 [0, 0, 1, 0]], dtype='complex_')


def test_process_tomography():
    # set operator circuit, num qubits
    cnot = catalog["klm cnot"].as_circuit().build()   # .build_circuit()

    p = pcvl.Processor("Naive", cnot)
    states = {
        pcvl.BasicState([1,0,1,0,0,1,0,1]): "00",
        pcvl.BasicState([1,0,0,1,0,1,0,1]): "01",
        pcvl.BasicState([0,1,1,0,0,1,0,1]): "10",
        pcvl.BasicState([0,1,0,1,0,1,0,1]): "11"
    }
    ca = pcvl.algorithm.Analyzer(p,states)
    pcvl.pdisplay(ca)

    herald = [(4, 0), (5, 1), (6, 0), (7, 1)]

    # create QST object
    qst = QuantumStateTomography(operator_circuit=cnot, nqubit=2, heralded_modes=herald)
    # create process tomography object and passing qst object as a parameter
    qpt = QuantumProcessTomography(nqubit=2, operator=operotor_to_test(), operator_circuit=cnot, qst=qst,
                                   heralded_modes=herald)

    # compute Chi matrix
    chi_op = qpt.chi_matrix()
    chi_op_ideal = qpt.chi_target(operotor_to_test())
    # Compute fidelity
    fidelity = FidelityTomography(nqubit=2)
    cnot_fidelity = fidelity.process_fidelity(chi_op, chi_op_ideal)
    assert cnot_fidelity == pytest.approx(1, 1e-3)  # computed fidelity is around 0.99967
