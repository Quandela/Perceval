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

import sys
from pathlib import Path

import pytest
import numpy as np
from perceval.components import catalog, Processor, BS
from perceval.algorithm import ProcessTomographyMLE, StateTomographyMLE
from perceval.algorithm.tomography.tomography_utils import process_fidelity, is_physical


CNOT_TARGET = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype='complex_')

GHZ_TARGET = np.zeros((8, 8))
GHZ_TARGET[0, 0], GHZ_TARGET[0, -1], GHZ_TARGET[-1, 0], GHZ_TARGET[-1, -1] = 1, 1, 1, 1
GHZ_TARGET /= 2

def fidelity_op_mle_process_tomography(op_proc):
    # create mle process tomography object
    qpt_mle = ProcessTomographyMLE(operator_processor=op_proc)
    chi_op = qpt_mle.chi_matrix()

    chi_op_ideal = qpt_mle.chi_target(CNOT_TARGET)

    op_fidelity = process_fidelity(chi_op, chi_op_ideal)
    return op_fidelity


def test_fidelity_heralded_cnot():
    cnot_p = catalog["heralded cnot"].build_processor()
    cnot_fidelity_mle = fidelity_op_mle_process_tomography(cnot_p)

    assert cnot_fidelity_mle == pytest.approx(1)

def test_ghz_state_tomography_mle():
    h_cnot_circ = catalog["klm cnot"].build_processor()

    ghz_state_proc = Processor("SLOS", 6)
    ghz_state_proc.add(0, BS.H())
    ghz_state_proc.add(0, h_cnot_circ)
    ghz_state_proc.add(2, h_cnot_circ)

    s_mle = StateTomographyMLE(ghz_state_proc)

    ghz_state = s_mle.state_tomography_density_matrix()

    fidelity = s_mle.state_fidelity(GHZ_TARGET, ghz_state)

    assert np.trace(ghz_state) == pytest.approx(1)
    assert fidelity == pytest.approx(1)

def test_chi_cnot_from_mle_is_physical():
    cnot_p = catalog["klm cnot"].build_processor()

    qpt = ProcessTomographyMLE(operator_processor=cnot_p)

    chi_op = qpt.chi_matrix()
    res = is_physical(chi_op, nqubit=2)

    assert res['Trace=1'] is True  # if Chi has Trace = 1
    assert res['Hermitian'] is True  # if Chi is Hermitian
    assert res['Completely Positive'] is True  # if input Chi is Completely Positive
