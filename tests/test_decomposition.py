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

import pytest
import perceval as pcvl
import perceval.lib.symb as symb
import perceval.lib.phys as phys

import numpy as np

TEST_DATA_DIR = Path(__file__).resolve().parent / 'data'


def test_perm_0():
    c = pcvl.Circuit(4).add(0, symb.PERM([3, 1, 2, 0]))
    ub = (pcvl.Circuit(2)
          // symb.BS()
          // (0, symb.PS(phi=pcvl.Parameter("φ_a")))
          // symb.BS()
          // (0, symb.PS(phi=pcvl.Parameter("φ_b"))))
    c1 = pcvl.Circuit.decomposition(pcvl.Matrix(c.U), ub, phase_shifter_fn=symb.PS, shape="triangle",
                                    permutation=symb.PERM)
    assert c1.describe().replace("\n", "").replace(" ", "") == """
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
    c1 = pcvl.Circuit.decomposition(pcvl.Matrix(c.U), ub, shape="triangle")
    m1 = c1.compute_unitary(use_symbolic=False)
    assert pytest.approx(1, rel=1e-3) == abs(m1[0][0])+1
    assert pytest.approx(2, rel=1e-3) == abs(m1[0][1])+1
    assert pytest.approx(2, rel=1e-3) == abs(m1[1][0])+1
    assert pytest.approx(1, rel=1e-3) == abs(m1[1][1])+1


def test_basic_perm_triangle_bs():
    c = pcvl.Circuit(2).add(0, symb.PERM([1, 0]))
    ub = (pcvl.Circuit(2)
          // symb.BS(theta=pcvl.Parameter("theta")))
    c1 = pcvl.Circuit.decomposition(pcvl.Matrix(c.U), ub, shape="triangle")
    m1 = c1.compute_unitary(use_symbolic=False)
    assert pytest.approx(1, rel=1e-3) == abs(m1[0][0])+1
    assert pytest.approx(2, rel=1e-3) == abs(m1[0][1])+1
    assert pytest.approx(2, rel=1e-3) == abs(m1[1][0])+1
    assert pytest.approx(1, rel=1e-3) == abs(m1[1][1])+1


@pytest.mark.skip(reason="rectangular decomposition not implemented")
def test_basic_perm_rectangle():
    c = pcvl.Circuit(2).add(0, symb.PERM([1, 0]))
    ub = (pcvl.Circuit(2)
          // (0, symb.PS(phi=pcvl.Parameter("φ_a")))
          // symb.BS()
          // (0, symb.PS(phi=pcvl.Parameter("φ_b")))
          // symb.BS())
    c1 = pcvl.Circuit.decomposition(pcvl.Matrix(c.U), ub, shape="rectangle")
    m1 = c1.compute_unitary(use_symbolic=False)
    assert pytest.approx(1, rel=1e-3) == abs(m1[0][0])+1
    assert pytest.approx(1, rel=1e-3) == abs(m1[0][1])
    assert pytest.approx(1, rel=1e-3) == abs(m1[1][0])
    assert pytest.approx(1, rel=1e-3) == abs(m1[1][1])+1


def test_perm_triangle():
    c = pcvl.Circuit(4).add(0, symb.PERM([3, 1, 2, 0]))
    ub = (pcvl.Circuit(2)
          // symb.BS()
          // (0, symb.PS(phi=pcvl.Parameter("φ_a")))
          // symb.BS()
          // (0, symb.PS(phi=pcvl.Parameter("φ_b"))))
    m = c.compute_unitary(False)
    c1 = pcvl.Circuit.decomposition(pcvl.Matrix(c.U), ub, shape="triangle")
    m1 = c1.compute_unitary(False)
    np.testing.assert_array_almost_equal(abs(m), abs(m1), decimal=6)


@pytest.mark.skip(reason="rectangular decomposition not implemented")
def test_perm_rectangle_bs_0():
    c = pcvl.Circuit(3).add(0, symb.PERM([1, 0, 2]))
    ub = (pcvl.Circuit(2)
          // (0, symb.PS(phi=pcvl.Parameter("φ_a")))
          // symb.BS(theta=pcvl.P("theta")))
    m = c.compute_unitary(False)
    c1 = pcvl.Circuit.decomposition(pcvl.Matrix(c.U), ub, shape="rectangle")
    m1 = c1.compute_unitary(False)
    np.testing.assert_array_almost_equal(abs(m), abs(m1), decimal=6)


@pytest.mark.skip(reason="rectangular decomposition not implemented")
def test_perm_rectangle_bs_1():
    c = pcvl.Circuit(3).add(0, symb.PERM([2, 1, 0]))
    ub = (pcvl.Circuit(2)
          // (0, symb.PS(phi=pcvl.Parameter("φ_a")))
          // symb.BS(theta=pcvl.P("theta")))
    m = c.compute_unitary(False)
    c1 = pcvl.Circuit.decomposition(pcvl.Matrix(c.U), ub, shape="rectangle", constraints=[(0, None)])
    m1 = c1.compute_unitary(False)
    np.testing.assert_array_almost_equal(abs(m), abs(m1), decimal=6)
    c2 = pcvl.Circuit.decomposition(pcvl.Matrix(c.U), ub, shape="rectangle", constraints=[(None, 0)])
    assert c2 is None


@pytest.mark.skip(reason="rectangular decomposition not implemented")
def test_id_decomposition_rectangle():
    # identity matrix decompose as ... identity
    c = pcvl.Circuit(4)
    ub = (pcvl.Circuit(2)
          // (0, symb.PS(phi=pcvl.Parameter("φ_a")))
          // symb.BS()
          // (0, symb.PS(phi=pcvl.Parameter("φ_b")))
          // symb.BS())
    c1 = pcvl.Circuit.decomposition(pcvl.Matrix(c.U), ub, shape="rectangle")
    np.testing.assert_array_almost_equal(pcvl.Matrix.eye(4, use_symbolic=False), c1.compute_unitary(False), decimal=6)


def test_id_decomposition_triangle():
    # identity matrix decompose as ... identity
    c = pcvl.Circuit(4)
    ub = (pcvl.Circuit(2)
          // symb.BS()
          // (0, symb.PS(phi=pcvl.Parameter("φ_a")))
          // symb.BS()
          // (0, symb.PS(phi=pcvl.Parameter("φ_b"))))
    c1 = pcvl.Circuit.decomposition(pcvl.Matrix(c.U), ub, shape="triangle")
    np.testing.assert_array_almost_equal(pcvl.Matrix.eye(4, use_symbolic=False), c1.compute_unitary(False), decimal=6)


def test_any_unitary_triangle():
    with open(TEST_DATA_DIR / 'u_random_3', "r") as f:
        m = pcvl.Matrix(f)
        ub = (pcvl.Circuit(2)
              // symb.BS()
              // (0, symb.PS(phi=pcvl.Parameter("φ_a")))
              // symb.BS()
              // (0, symb.PS(phi=pcvl.Parameter("φ_b"))))
        c1 = pcvl.Circuit.decomposition(m, ub, phase_shifter_fn=symb.PS, shape="triangle", max_try=10)
        assert c1 is not None
        np.testing.assert_array_almost_equal(m, c1.compute_unitary(False), decimal=6)


def test_any_unitary_triangle_bad_ud():
    with open(TEST_DATA_DIR / 'u_random_3', "r") as f:
        m = pcvl.Matrix(f)
        ub = (pcvl.Circuit(2)
              // (0, symb.PS(phi=pcvl.Parameter("φ_a")))
              // symb.BS()
              // (0, symb.PS(phi=pcvl.Parameter("φ_b")))
              // symb.BS())
        c1 = pcvl.Circuit.decomposition(m, ub, phase_shifter_fn=symb.PS, shape="triangle", max_try=10)
        assert c1 is None


@pytest.mark.skip(reason="rectangular decomposition not implemented")
def test_any_unitary_rectangle():
    with open(TEST_DATA_DIR / 'u_random_3', "r") as f:
        m = pcvl.Matrix(f)
        ub = (pcvl.Circuit(2)
              // (0, symb.PS(phi=pcvl.Parameter("φ_a")))
              // symb.BS()
              // (0, symb.PS(phi=pcvl.Parameter("φ_b")))
              // symb.BS())
        c1 = pcvl.Circuit.decomposition(m, ub, phase_shifter_fn=symb.PS, shape="rectangle", max_try=10)
        assert c1 is not None
        np.testing.assert_array_almost_equal(m, c1.compute_unitary(False), decimal=6)


def test_simple_phase():
    for m in [pcvl.Matrix([[0, 1j], [1, 0]]), pcvl.Matrix([[0, 1j], [-1, 0]]), pcvl.Matrix([[1j, 0], [0, -1]])]:
        ub = (pcvl.Circuit(2)
              // symb.BS()
              // (0, symb.PS(phi=pcvl.Parameter("φ_b")))
              // symb.BS()
              // (0, symb.PS(phi=pcvl.Parameter("φ_a"))))
        c1 = pcvl.Circuit.decomposition(m, ub, phase_shifter_fn=symb.PS, shape="triangle", max_try=5)
        assert c1 is not None
        np.testing.assert_array_almost_equal(m, c1.compute_unitary(False), decimal=6)


def test_decompose_non_unitary():
    m = np.array([[(i and j) and (i+j*1j)/np.sqrt(i*i+j*j) or 0 for i in range(5)] for j in range(5)])
    ub = (pcvl.Circuit(2)
          // symb.BS()
          // (0, symb.PS(phi=pcvl.Parameter("φ_b")))
          // symb.BS()
          // (0, symb.PS(phi=pcvl.Parameter("φ_a"))))
    with pytest.raises(ValueError):
        pcvl.Circuit.decomposition(m, ub, shape="triangle", max_try=5)


def test_decomposition_large():
    with open(TEST_DATA_DIR / 'u_random_8', "r") as f:
        m = pcvl.Matrix(f)
        ub = (pcvl.Circuit(2)
              // symb.BS()
              // (0, symb.PS(phi=pcvl.Parameter("φ_a")))
              // symb.BS()
              // (0, symb.PS(phi=pcvl.Parameter("φ_b"))))
        c1 = pcvl.Circuit.decomposition(m, ub, phase_shifter_fn=symb.PS, shape="triangle", max_try=1)
        assert c1 is not None
        np.testing.assert_array_almost_equal(m, c1.compute_unitary(False), decimal=6)


def test_decomposition_perm():
    c1 = pcvl.Circuit.decomposition(pcvl.Matrix(symb.PERM([3, 1, 0, 2]).U), symb.BS(R=pcvl.P("R")),
                                    phase_shifter_fn=symb.PS)
    assert c1 is not None


def test_decomposition_inverse_symb():
    ub = (symb.Circuit(2)
          // (0, symb.BS(theta=pcvl.Parameter("theta")))
          // (1, symb.PS(phi=pcvl.Parameter("phi"))))

    u = pcvl.Matrix.random_unitary(6)
    c = pcvl.Circuit.decomposition(u,
                                   ub,
                                   inverse_v=True,
                                   inverse_h=True,
                                   phase_shifter_fn=symb.PS)
    np.testing.assert_array_almost_equal(u, c.compute_unitary(False), decimal=6)


def test_decomposition_inverse_phys():
    ub = (phys.Circuit(2)
          // (0, phys.BS(theta=pcvl.Parameter("theta")))
          // (1, phys.PS(phi=pcvl.Parameter("phi"))))

    u = pcvl.Matrix.random_unitary(6)
    c = pcvl.Circuit.decomposition(u,
                                   ub,
                                   inverse_v=True,
                                   inverse_h=True,
                                   phase_shifter_fn=phys.PS)
    np.testing.assert_array_almost_equal(u, c.compute_unitary(False), decimal=6)
