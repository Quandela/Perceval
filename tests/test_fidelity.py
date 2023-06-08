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
import perceval.utils.algorithms.norm as norm


def test_fidelity():
    size = 8

    # fidelity of a matrix with itself is maximum (1)
    random_unitary = pcvl.Matrix.random_unitary(size)
    assert norm.fidelity(random_unitary, random_unitary) == pytest.approx(1)

    # fidelity is commutative
    random_unitary_2 = pcvl.Matrix.random_unitary(size)
    assert norm.fidelity(random_unitary, random_unitary_2) == \
           pytest.approx(norm.fidelity(random_unitary_2, random_unitary))

    # fidelity of orthognonal matrices is minimum (0)
    identity = pcvl.Matrix(np.identity(size))
    flipped = pcvl.Matrix(np.flipud(identity))
    assert norm.fidelity(identity, flipped) == pytest.approx(0)

    # fidelity of any matrix with zeros is 0
    zero_mat = pcvl.Matrix(np.zeros((size, size)))
    assert norm.fidelity(random_unitary, zero_mat) == pytest.approx(0)

    # fidelity can be computed only for same size matrices
    mat_too_small = pcvl.Matrix.random_unitary(size // 2)
    with pytest.raises(ValueError):
        norm.fidelity(random_unitary, mat_too_small)
