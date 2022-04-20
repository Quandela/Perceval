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
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from pathlib import Path
import random

from pytest import approx
import perceval as pcvl
import perceval.lib.symb as symb

import numpy as np

TEST_DATA_DIR = Path(__file__).resolve().parent / 'data'

def test_perm_0():
    c = pcvl.Circuit(4).add(0, symb.PERM([3, 1, 2, 0]))
    ub = (pcvl.Circuit(2)
          // symb.BS()
          // (0, symb.PS(phi=pcvl.Parameter("φ_a")))
          // symb.BS()
          // (0, symb.PS(phi=pcvl.Parameter("φ_b"))))
    C1 = pcvl.Circuit.decomposition(pcvl.Matrix(c.U), ub, phase_shifter_fn=symb.PS, shape="triangle",
                                    permutation=symb.PERM)
    assert C1.describe().replace("\n", "").replace(" ", "") == """
        Circuit(4).add([0, 1], symb.PERM([1, 0]))
                  .add([1, 2], symb.PERM([1, 0]))
                  .add([2, 3], symb.PERM([1, 0]))
                  .add([1, 2], symb.PERM([1, 0]))
                  .add([0, 1], symb.PERM([1, 0]))""".replace("\n", "").replace(" ", "")


def test_basic_perm_triangle():
    c = pcvl.Circuit(2).add(0, symb.PERM([1, 0]))
    ub = (pcvl.Circuit(2)
          // symb.BS()
          // (0, symb.PS(phi=pcvl.Parameter("φ_a")))
          // symb.BS()
          // (0, symb.PS(phi=pcvl.Parameter("φ_b"))))
    C1 = pcvl.Circuit.decomposition(pcvl.Matrix(c.U), ub, shape="triangle")
    M1 = C1.compute_unitary(use_symbolic=False)
    assert approx(1, rel=1e-3) == abs(M1[0][0])+1
    assert approx(2, rel=1e-3) == abs(M1[0][1])+1
    assert approx(2, rel=1e-3) == abs(M1[1][0])+1
    assert approx(1, rel=1e-3) == abs(M1[1][1])+1


def test_basic_perm_triangle_bs():
    c = pcvl.Circuit(2).add(0, symb.PERM([1, 0]))
    ub = (pcvl.Circuit(2)
          // symb.BS(theta=pcvl.Parameter("theta")))
    C1 = pcvl.Circuit.decomposition(pcvl.Matrix(c.U), ub, shape="triangle")
    M1 = C1.compute_unitary(use_symbolic=False)
    assert approx(1, rel=1e-3) == abs(M1[0][0])+1
    assert approx(2, rel=1e-3) == abs(M1[0][1])+1
    assert approx(2, rel=1e-3) == abs(M1[1][0])+1
    assert approx(1, rel=1e-3) == abs(M1[1][1])+1


def test_basic_perm_rectangle():
    c = pcvl.Circuit(2).add(0, symb.PERM([1, 0]))
    ub = (pcvl.Circuit(2)
          // (0, symb.PS(phi=pcvl.Parameter("φ_a")))
          // symb.BS()
          // (0, symb.PS(phi=pcvl.Parameter("φ_b")))
          // symb.BS())
    C1 = pcvl.Circuit.decomposition(pcvl.Matrix(c.U), ub, shape="rectangle")
    M1 = C1.compute_unitary(use_symbolic=False)
    assert approx(1, rel=1e-3) == abs(M1[0][0])+1
    assert approx(1, rel=1e-3) == abs(M1[0][1])
    assert approx(1, rel=1e-3) == abs(M1[1][0])
    assert approx(1, rel=1e-3) == abs(M1[1][1])+1


def test_perm_triangle():
    c = pcvl.Circuit(4).add(0, symb.PERM([3, 1, 2, 0]))
    ub = (pcvl.Circuit(2)
          // symb.BS()
          // (0, symb.PS(phi=pcvl.Parameter("φ_a")))
          // symb.BS()
          // (0, symb.PS(phi=pcvl.Parameter("φ_b"))))
    M = c.compute_unitary(False)
    C1 = pcvl.Circuit.decomposition(pcvl.Matrix(c.U), ub, shape="triangle")
    M1 = C1.compute_unitary(False)
    np.testing.assert_array_almost_equal(abs(M), abs(M1), decimal=6)


def test_perm_rectangle_bs_0():
    c = pcvl.Circuit(3).add(0, symb.PERM([1, 0, 2]))
    ub = (pcvl.Circuit(2)
          // (0, symb.PS(phi=pcvl.Parameter("φ_a")))
          // symb.BS(theta=pcvl.P("theta")))
    M = c.compute_unitary(False)
    C1 = pcvl.Circuit.decomposition(pcvl.Matrix(c.U), ub, shape="rectangle")
    M1 = C1.compute_unitary(False)
    np.testing.assert_array_almost_equal(abs(M), abs(M1), decimal=6)


def test_perm_rectangle_bs_1():
    c = pcvl.Circuit(3).add(0, symb.PERM([2, 1, 0]))
    ub = (pcvl.Circuit(2)
          // (0, symb.PS(phi=pcvl.Parameter("φ_a")))
          // symb.BS(theta=pcvl.P("theta")))
    M = c.compute_unitary(False)
    C1 = pcvl.Circuit.decomposition(pcvl.Matrix(c.U), ub, shape="rectangle", constraints=[(0, None)])
    M1 = C1.compute_unitary(False)
    np.testing.assert_array_almost_equal(abs(M), abs(M1), decimal=6)
    C2 = pcvl.Circuit.decomposition(pcvl.Matrix(c.U), ub, shape="rectangle", constraints=[(None, 0)])
    assert C2 is None

def test_id_decomposition_rectangle():
    # identity matrix decompose as ... identity
    c = pcvl.Circuit(4)
    ub = (pcvl.Circuit(2)
          // (0, symb.PS(phi=pcvl.Parameter("φ_a")))
          // symb.BS()
          // (0, symb.PS(phi=pcvl.Parameter("φ_b")))
          // symb.BS())
    C1 = pcvl.Circuit.decomposition(pcvl.Matrix(c.U), ub, shape="rectangle")
    np.testing.assert_array_almost_equal(pcvl.Matrix.eye(4, use_symbolic=False), C1.compute_unitary(False), decimal=6)

def test_id_decomposition_triangle():
    # identity matrix decompose as ... identity
    c = pcvl.Circuit(4)
    ub = (pcvl.Circuit(2)
          // symb.BS()
          // (0, symb.PS(phi=pcvl.Parameter("φ_a")))
          // symb.BS()
          // (0, symb.PS(phi=pcvl.Parameter("φ_b"))))
    C1 = pcvl.Circuit.decomposition(pcvl.Matrix(c.U), ub, shape="triangle")
    np.testing.assert_array_almost_equal(pcvl.Matrix.eye(4, use_symbolic=False), C1.compute_unitary(False), decimal=6)


def test_any_unitary_triangle():
    with open(TEST_DATA_DIR / 'u_random_3', "r") as f:
        M = pcvl.Matrix(f)
        ub = (pcvl.Circuit(2)
              // symb.BS()
              // (0, symb.PS(phi=pcvl.Parameter("φ_a")))
              // symb.BS()
              // (0, symb.PS(phi=pcvl.Parameter("φ_b"))))
        C1 = pcvl.Circuit.decomposition(M, ub, phase_shifter_fn=symb.PS, shape="triangle", max_try=5)
        assert C1 is not None
        np.testing.assert_array_almost_equal(M, C1.compute_unitary(False), decimal=6)


def test_any_unitary_triangle_bad_ud():
    with open(TEST_DATA_DIR / 'u_random_3', "r") as f:
        M = pcvl.Matrix(f)
        ub = (pcvl.Circuit(2)
              // (0, symb.PS(phi=pcvl.Parameter("φ_a")))
              // symb.BS()
              // (0, symb.PS(phi=pcvl.Parameter("φ_b")))
              // symb.BS())
        C1 = pcvl.Circuit.decomposition(M, ub, phase_shifter_fn=symb.PS, shape="triangle", max_try=10)
        assert C1 is None


def test_any_unitary_rectangle():
    with open(TEST_DATA_DIR / 'u_random_3', "r") as f:
        M = pcvl.Matrix(f)
        ub = (pcvl.Circuit(2)
              // (0, symb.PS(phi=pcvl.Parameter("φ_a")))
              // symb.BS()
              // (0, symb.PS(phi=pcvl.Parameter("φ_b")))
              // symb.BS())
        C1 = pcvl.Circuit.decomposition(M, ub, phase_shifter_fn=symb.PS, shape="rectangle", max_try=10)
        assert C1 is not None
        np.testing.assert_array_almost_equal(M, C1.compute_unitary(False), decimal=6)


def test_simple_phase():
    for M in [pcvl.Matrix([[0, 1j], [1, 0]]), pcvl.Matrix([[0, 1j], [-1, 0]]), pcvl.Matrix([[1j, 0], [0, -1]])]:
        ub = (pcvl.Circuit(2)
              // symb.BS()
              // (0, symb.PS(phi=pcvl.Parameter("φ_b")))
              // symb.BS()
              // (0, symb.PS(phi=pcvl.Parameter("φ_a"))))
        C1 = pcvl.Circuit.decomposition(M, ub, phase_shifter_fn=symb.PS, shape="triangle", max_try=10)
        assert C1 is not None
        np.testing.assert_array_almost_equal(M, C1.compute_unitary(False), decimal=6)


def test_decompose_non_unitary():
    M = np.array([[random.random() for i in range(5)] for j in range(5)])
    ub = (pcvl.Circuit(2)
          // symb.BS()
          // (0, symb.PS(phi=pcvl.Parameter("φ_b")))
          // symb.BS()
          // (0, symb.PS(phi=pcvl.Parameter("φ_a"))))
    C1 = pcvl.Circuit.decomposition(M, ub, shape="triangle", max_try=10)
    assert C1 is None, "should not be able to decompose a non unitary matrix"
