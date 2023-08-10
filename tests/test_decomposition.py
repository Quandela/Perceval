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

from pathlib import Path
import pytest
import perceval as pcvl
import perceval.components as comp
from perceval.utils.algorithms.circuit_optimizer import CircuitOptimizer
from perceval.utils.algorithms import norm

import numpy as np

TEST_DATA_DIR = Path(__file__).resolve().parent / 'data'


def test_perm_0():
    c = pcvl.Circuit(4).add(0, comp.PERM([3, 1, 2, 0]))
    c1 = pcvl.Circuit.decomposition(pcvl.Matrix(c.U), pcvl.catalog["mzi phase last"].build_circuit(),
                                    phase_shifter_fn=comp.PS, shape="triangle", permutation=comp.PERM)
    assert c1.describe().replace("\n", "").replace(" ", "") == """
        Circuit(4).add((0, 1), PERM([1, 0]))
                  .add((1, 2), PERM([1, 0]))
                  .add((2, 3), PERM([1, 0]))
                  .add((1, 2), PERM([1, 0]))
                  .add((0, 1), PERM([1, 0]))""".replace("\n", "").replace(" ", "")


def test_basic_perm_triangle():
    c = pcvl.Circuit(2).add(0, comp.PERM([1, 0]))
    c1 = pcvl.Circuit.decomposition(pcvl.Matrix(c.U), pcvl.catalog["mzi phase last"].build_circuit(), shape="triangle")
    m1 = c1.compute_unitary(use_symbolic=False)
    assert pytest.approx(0, abs=1e-7) == abs(m1[0][0])
    assert pytest.approx(1, abs=1e-7) == abs(m1[0][1])
    assert pytest.approx(1, abs=1e-7) == abs(m1[1][0])
    assert pytest.approx(0, abs=1e-7) == abs(m1[1][1])


def test_basic_perm_triangle_bs():
    c = pcvl.Circuit(2).add(0, comp.PERM([1, 0]))
    ub = comp.BS(theta=pcvl.Parameter("theta"))
    c1 = pcvl.Circuit.decomposition(pcvl.Matrix(c.U), ub, shape="triangle")
    m1 = c1.compute_unitary(use_symbolic=False)
    assert pytest.approx(0, abs=1e-7) == abs(m1[0][0])
    assert pytest.approx(1, abs=1e-7) == abs(m1[0][1])
    assert pytest.approx(1, abs=1e-7) == abs(m1[1][0])
    assert pytest.approx(0, abs=1e-7) == abs(m1[1][1])


def test_basic_perm_rectangle():
    c = pcvl.Circuit(2).add(0, comp.PERM([1, 0]))
    co = CircuitOptimizer()
    c1 = co.optimize_rectangle(pcvl.Matrix(c.U))
    m1 = c1.compute_unitary()
    assert pytest.approx(1, rel=1e-3) == abs(m1[0][0])+1
    assert pytest.approx(1, rel=1e-3) == abs(m1[0][1])
    assert pytest.approx(1, rel=1e-3) == abs(m1[1][0])
    assert pytest.approx(1, rel=1e-3) == abs(m1[1][1])+1


def test_perm_triangle():
    c = pcvl.Circuit(4).add(0, comp.PERM([3, 1, 2, 0]))
    m = c.compute_unitary()
    c1 = pcvl.Circuit.decomposition(pcvl.Matrix(c.U), pcvl.catalog["mzi phase last"].build_circuit(), shape="triangle")
    m1 = c1.compute_unitary()
    np.testing.assert_array_almost_equal(abs(m), abs(m1), decimal=6)

@pytest.mark.skip(reason="Optimization does not converge with unusual template component")
def test_perm_rectangle_bs_0():
    c = pcvl.Circuit(3).add(0, comp.PERM([1, 0, 2]))
    def gen_template_component(i: int):
        return (pcvl.Circuit(2)
                // comp.PS(phi=pcvl.Parameter(f"phi_a{i}"))
                // comp.BS(theta=pcvl.Parameter(f"theta{i}")))
    co = CircuitOptimizer()
    co.threshold = 0.1
    c1 = co.optimize_rectangle(pcvl.Matrix(c.U), gen_template_component, phase_at_output=True)
    assert norm.fidelity(c.compute_unitary(), c1.compute_unitary()) > 1 - co.threshold


@pytest.mark.skip(reason="Optimization does not converge with unusual template component")
def test_perm_rectangle_bs_1():
    c = pcvl.Circuit(3).add(0, comp.PERM([2, 1, 0]))
    ub = (pcvl.Circuit(2)
          // (0, comp.PS(phi=pcvl.Parameter("φ_a")))
          // comp.BS(theta=pcvl.P("theta")))
    m = c.compute_unitary(False)
    c1 = pcvl.Circuit.decomposition(pcvl.Matrix(c.U), ub, shape="rectangle", constraints=[(0, None)])
    m1 = c1.compute_unitary(False)
    np.testing.assert_array_almost_equal(abs(m), abs(m1), decimal=6)
    c2 = pcvl.Circuit.decomposition(pcvl.Matrix(c.U), ub, shape="rectangle", constraints=[(None, 0)])
    assert c2 is None


def test_id_decomposition_rectangle():
    # identity matrix decompose as ... identity
    c = pcvl.Circuit(4)
    co = CircuitOptimizer()
    c1 = co.optimize_rectangle(pcvl.Matrix(c.U), pcvl.catalog["mzi phase first"].generate, phase_at_output=True)
    m1 = c1.compute_unitary(False)
    assert norm.fidelity(c.compute_unitary(), m1) > 1 - co.threshold


def test_id_decomposition_triangle():
    # identity matrix decompose as ... identity
    c = pcvl.Circuit(4)
    c1 = pcvl.Circuit.decomposition(pcvl.Matrix(c.U), pcvl.catalog["mzi phase last"].build_circuit(), shape="triangle")
    np.testing.assert_array_almost_equal(pcvl.Matrix.eye(4, use_symbolic=False), c1.compute_unitary(False), decimal=6)
    assert c1.ncomponents() == 0

    # With ignore_identity_block=False, the decomposed circuit contains multiple MZIs with trivial values
    c2 = pcvl.Circuit.decomposition(pcvl.Matrix(c.U), pcvl.catalog["mzi phase last"].build_circuit(), shape="triangle",
                                    ignore_identity_block=False)
    assert c2 is not None
    m2 = c2.compute_unitary()
    assert pytest.approx(1, abs=1e-7) == abs(m2[0][0])
    assert pytest.approx(0, abs=1e-7) == abs(m2[0][1])
    assert pytest.approx(0, abs=1e-7) == abs(m2[1][0])
    assert pytest.approx(1, abs=1e-7) == abs(m2[1][1])
    assert c2.ncomponents() == 6*pcvl.catalog["mzi phase last"].build_circuit().ncomponents()


def test_any_unitary_triangle():
    with open(TEST_DATA_DIR / 'u_random_3', "r") as f:
        m = pcvl.Matrix(f)
        c1 = pcvl.Circuit.decomposition(m, pcvl.catalog["mzi phase last"].build_circuit(), phase_shifter_fn=comp.PS,
                                        shape="triangle", max_try=10)
        assert c1 is not None
        np.testing.assert_array_almost_equal(m, c1.compute_unitary(), decimal=6)


def test_any_unitary_rectangle():
    with open(TEST_DATA_DIR / 'u_random_8', "r") as f:
        m = pcvl.Matrix(f)
        co = CircuitOptimizer()
        c1 = co.optimize_rectangle(m)
        m1 = c1.compute_unitary()
        assert norm.fidelity(m, m1) > 1 - co.threshold

        # You can decompose with another form of MZI as long as your template remains universal
        # In this case, that means putting a layer of PS at the input of the circuit
        c2 = co.optimize_rectangle(m, pcvl.catalog["mzi phase last"].generate, phase_at_output=False)
        m2 = c2.compute_unitary()
        assert norm.fidelity(m, m2) > 1 - co.threshold


def test_simple_phase():
    for m in [pcvl.Matrix([[0, 1j], [1, 0]]), pcvl.Matrix([[0, 1j], [-1, 0]]), pcvl.Matrix([[1j, 0], [0, -1]])]:
        c1 = pcvl.Circuit.decomposition(m, pcvl.catalog["mzi phase last"].build_circuit(), phase_shifter_fn=comp.PS,
                                        shape="triangle", max_try=5)
        assert c1 is not None
        np.testing.assert_array_almost_equal(m, c1.compute_unitary(False), decimal=6)


def test_decompose_non_unitary():
    m = np.array([[(i and j) and (i+j*1j)/np.sqrt(i*i+j*j) or 0 for i in range(5)] for j in range(5)])
    with pytest.raises(ValueError):
        pcvl.Circuit.decomposition(m, pcvl.catalog["mzi phase last"].build_circuit(), shape="triangle", max_try=5)


def test_decomposition_large():
    with open(TEST_DATA_DIR / 'u_random_8', "r") as f:
        m = pcvl.Matrix(f)
        c1 = pcvl.Circuit.decomposition(m, pcvl.catalog["mzi phase last"].build_circuit(), phase_shifter_fn=comp.PS,
                                        shape="triangle", max_try=1)
        assert c1 is not None
        np.testing.assert_array_almost_equal(m, c1.compute_unitary(False), decimal=6)


def test_decomposition_perm():
    c1 = pcvl.Circuit.decomposition(pcvl.Matrix(comp.PERM([3, 1, 0, 2]).U), comp.BS(theta=pcvl.P("theta")),
                                    phase_shifter_fn=comp.PS)
    assert c1 is not None


def test_decomposition_inverse_rx():
    ub = (pcvl.Circuit(2)
          // (0, comp.BS.Rx(theta=pcvl.Parameter("theta")))
          // (1, comp.PS(phi=pcvl.Parameter("phi"))))

    u = pcvl.Matrix.random_unitary(6)
    c = pcvl.Circuit.decomposition(u,
                                   ub,
                                   inverse_v=True,
                                   inverse_h=True,
                                   phase_shifter_fn=comp.PS)
    np.testing.assert_array_almost_equal(u, c.compute_unitary(False), decimal=6)


def test_decomposition_inverse_h():
    ub = (pcvl.Circuit(2)
          // (0, comp.BS.H(theta=pcvl.Parameter("theta")))
          // (1, comp.PS(phi=pcvl.Parameter("phi"))))

    u = pcvl.Matrix.random_unitary(6)
    c = pcvl.Circuit.decomposition(u,
                                   ub,
                                   inverse_v=True,
                                   inverse_h=True,
                                   phase_shifter_fn=comp.PS)
    np.testing.assert_array_almost_equal(u, c.compute_unitary(False), decimal=6)
