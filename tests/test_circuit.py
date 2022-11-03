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

import pytest
from pathlib import Path

from perceval import Circuit, P, BasicState, pdisplay, Matrix, BackendFactory, Processor
from perceval.rendering.pdisplay import pdisplay_circuit, pdisplay_matrix, pdisplay_analyzer
from perceval.rendering.format import Format
import perceval.algorithm as algo
import perceval.components.unitary_components as comp
import sympy as sp
import numpy as np


def strip_line_12(s: str) -> str:
    return s.strip().replace("            ", "")


def test_helloword():
    c = comp.BS()
    assert c.m == 2
    definition = c.definition()
    assert strip_line_12(pdisplay_matrix(definition)) == strip_line_12("""
            ⎡exp(I*(phi_tl + phi_tr))*cos(theta/2)    I*exp(I*(phi_bl + phi_tr))*sin(theta/2)⎤
            ⎣I*exp(I*(phi_br + phi_tl))*sin(theta/2)  exp(I*(phi_bl + phi_br))*cos(theta/2)  ⎦
    """)
    expected = [P(name='theta', value=sp.pi/2, min_v=0, max_v=4*sp.pi),
                P(name='phi_tl', value=0, min_v=0, max_v=2*sp.pi),
                P(name='phi_bl', value=0, min_v=0, max_v=2*sp.pi),
                P(name='phi_tr', value=0, min_v=0, max_v=2*sp.pi),
                P(name='phi_br', value=0, min_v=0, max_v=2*sp.pi),
                ]
    for p_res, p_exp in zip(c.get_parameters(True), expected):
        assert str(p_res) == str(p_exp)
    assert c.U.is_unitary()
    for backend_name in ["SLOS", "Naive"]:
        backend = BackendFactory.get_backend(backend_name)
        simulator = backend(c.U)
        expected_outputs = {
            BasicState("|0,1>"): 0.5,
            BasicState("|1,0>"): 0.5
        }
        input_state = BasicState("|0,1>")
        count = 0
        for output_state in simulator.allstate_iterator(input_state):
            assert output_state in expected_outputs
            assert pytest.approx(expected_outputs[output_state]) == simulator.prob(input_state, output_state)
            count += 1
        assert count == len(expected_outputs)
        p = Processor(backend_name, c)
        ca = algo.Analyzer(p,
                           [BasicState([0, 1]), BasicState([1, 0]), BasicState([1, 1])],  # the input states
                           "*"  # all possible output states that can be generated with 1 or 2 photons
                           )
        assert strip_line_12(pdisplay_analyzer(ca)) == strip_line_12("""
            +-------+-------+-------+-------+-------+-------+
            |       | |0,1> | |1,0> | |2,0> | |1,1> | |0,2> |
            +-------+-------+-------+-------+-------+-------+
            | |0,1> |  1/2  |  1/2  |   0   |   0   |   0   |
            | |1,0> |  1/2  |  1/2  |   0   |   0   |   0   |
            | |1,1> |   0   |   0   |  1/2  |   0   |  1/2  |
            +-------+-------+-------+-------+-------+-------+
        """)


def test_empty_circuit():
    c = Circuit(4)
    m = c.compute_unitary(False)
    assert m.shape == (4, 4)
    assert np.allclose(m, Matrix.eye(4))
    assert pdisplay_circuit(c).replace(" ", "") == """

0:────:0 (depth 0)


1:────:1 (depth 0)


2:────:2 (depth 0)


3:────:3 (depth 0)

""".replace(" ", "")


def test_bs_symbolic_unitary():
    phi_tl = P("phi")
    theta = P("theta")
    bs = comp.BS(theta=theta, phi_tl=phi_tl)
    assert strip_line_12(pdisplay_matrix(bs.compute_unitary(use_symbolic=True))) == strip_line_12("""
            ⎡exp(I*phi)*cos(theta/2)    I*sin(theta/2)⎤
            ⎣I*exp(I*phi)*sin(theta/2)  cos(theta/2)  ⎦""")


def test_bs():
    bs = comp.BS()
    assert pdisplay_matrix(bs.U) == "⎡sqrt(2)/2    sqrt(2)*I/2⎤\n⎣sqrt(2)*I/2  sqrt(2)/2  ⎦"
    for backend_name in ["SLOS", "Naive"]:
        backend = BackendFactory.get_backend(backend_name)
        sbs = backend(bs.U)
        for _ in range(10):
            out = sbs.sample(BasicState("|0,1>"))
            assert str(out) == "|0,1>" or str(out) == "|1,0>"
        p = Processor(backend_name, bs)
        ca = algo.Analyzer(p, [BasicState([0, 1]), BasicState([1, 0])])
        ca.compute()
        assert pdisplay_analyzer(ca, nsimplify=True) == strip_line_12("""
            +-------+-------+-------+
            |       | |0,1> | |1,0> |
            +-------+-------+-------+
            | |0,1> |  1/2  |  1/2  |
            | |1,0> |  1/2  |  1/2  |
            +-------+-------+-------+""")
        assert pdisplay_analyzer(ca, nsimplify=False) == strip_line_12("""
            +-------+-------+-------+
            |       | |0,1> | |1,0> |
            +-------+-------+-------+
            | |0,1> |  0.5  |  0.5  |
            | |1,0> |  0.5  |  0.5  |
            +-------+-------+-------+""")


def test_bs_0():
    bs = comp.BS(theta=comp.BS.r_to_theta(1))
    assert pdisplay_matrix(bs.U) == "⎡1  0⎤\n⎣0  1⎦"
    for backend in ["SLOS", "Naive"]:
        simulator_backend = BackendFactory().get_backend(backend)
        sbs = simulator_backend(bs.U)
        for _ in range(10):
            out = sbs.sample(BasicState("|0,1>"))
            assert str(out) == "|0,1>"


def test_sbs_1():
    bs = comp.BS(theta=comp.BS.r_to_theta(0))
    assert pdisplay_matrix(bs.U) == "⎡0  I⎤\n⎣I  0⎦"
    for backend in ["SLOS", "Naive"]:
        simulator_backend = BackendFactory().get_backend(backend)
        sbs = simulator_backend(bs.U)
        for _ in range(10):
            out = sbs.sample(BasicState("|0,1>"))
            assert str(out) == "|1,0>"


def test_parameter():
    theta = P("theta0")
    bs = comp.BS(theta=theta)
    with pytest.raises(AssertionError):  # Cannot request a numeric matrix with variable parameters
        bs.compute_unitary(use_symbolic=False)
    assert pdisplay_matrix(bs.compute_unitary(use_symbolic=True)) == strip_line_12("""
            ⎡cos(theta0/2)    I*sin(theta0/2)⎤
            ⎣I*sin(theta0/2)  cos(theta0/2)  ⎦""")


def test_double_parameter_ok():
    phi1 = P("phi")
    comp.BS(phi_tl=phi1, phi_bl=phi1)


def test_double_parameter_dup():
    phi1 = P("phi")
    phi2 = P("phi")
    with pytest.raises(RuntimeError):  # Exception should have been generated for two parameters with same name
        comp.BS(phi_tr=phi1, phi_bl=phi2)


def test_double_parameter_dup_multi():
    phi1 = P("phi")
    phi2 = P("phi")
    with pytest.raises(RuntimeError):  # Exception should have been generated for two parameters with same name
        comp.BS(phi_tr=phi1) // comp.BS(phi_tl=phi2)


def test_build_composition():
    a = comp.BS()
    b = comp.BS()
    c = a // b
    assert pdisplay_matrix(c.U) == "⎡0  I⎤\n⎣I  0⎦"


def test_build_composition_2():
    c = comp.BS() // comp.PS(phi=sp.pi/2)
    assert pdisplay_matrix(c.U) == "⎡sqrt(2)*I/2  -sqrt(2)/2⎤\n⎣sqrt(2)*I/2  sqrt(2)/2 ⎦"


def test_build_composition_3():
    c = comp.BS() // (0, comp.PS(phi=sp.pi/2))
    assert pdisplay_matrix(c.U) == "⎡sqrt(2)*I/2  -sqrt(2)/2⎤\n⎣sqrt(2)*I/2  sqrt(2)/2 ⎦"


def test_build_composition_4():
    c = comp.BS() // (1, comp.PS(phi=sp.pi/2))
    assert pdisplay_matrix(c.U) == "⎡sqrt(2)/2   sqrt(2)*I/2⎤\n⎣-sqrt(2)/2  sqrt(2)*I/2⎦"


def test_invalid_ifloor():
    with pytest.raises(AssertionError):
        comp.BS() // (1, comp.BS())  # invalid ifloor should fail


def test_unitary_component():
    non_unitary_matrix = Matrix([[1, 2], [3, 4]])
    with pytest.raises(AssertionError):
        comp.Unitary(non_unitary_matrix)

    odd_size_matrix = Matrix.random_unitary(5)
    with pytest.raises(AssertionError):
        # In case the unitary component is polarized, the unitary matrix size must be even
        comp.Unitary(odd_size_matrix, use_polarization=True)

    unitary = comp.Unitary(odd_size_matrix)
    assert (unitary.U == odd_size_matrix).all()


def test_unitary_inverse():
    """
    Testing vertical inversion can be performed by applying a m:-1:0 permutation on the right and on the left of the
    inverted circuit. The resulting circuit has to be equivalent to the input circuit.
    """
    size = 4
    perm_tester = comp.PERM(list(range(size-1, -1, -1)))
    input_component = comp.Unitary(Matrix.random_unitary(size))
    inverted_component = comp.Unitary(input_component.U)
    inverted_component.inverse(v=True)
    test_circuit = Circuit(size)\
        .add(0, perm_tester)\
        .add(0, inverted_component)\
        .add(0, perm_tester)
    assert np.array_equal(input_component.U, test_circuit.U)

    # Test v and h inversion interaction (the order should have no impact on the result)
    u1 = comp.Unitary(Matrix.random_unitary(size))
    u2 = comp.Unitary(U=u1.U)
    u1.inverse(h=True)
    u1.inverse(v=True)
    u2.inverse(v=True)
    u2.inverse(h=True)
    assert np.allclose(u1.U, u2.U, atol=1e-12)


def _gen_bs(i: int):
    return comp.BS(theta=P(f"theta{i}"))


# noinspection PyTypeChecker
def test_generator():
    c = Circuit.generic_interferometer(5, _gen_bs)
    assert len(c.get_parameters()) == 5*4/2
    c = Circuit.generic_interferometer(5, _gen_bs, depth=1)
    assert len(c.get_parameters()) == 2
    c = Circuit.generic_interferometer(5, _gen_bs, depth=2)
    assert len(c.get_parameters()) == 4


def test_iterator():
    c = Circuit(3)
    comps = [(0, 1), (1, 2), (0, 1)]
    for k in range(len(comps)):
        c.add(comps[k], comp.BS(theta=1/(k+1)))

    d = Circuit(4)
    d.add((0, 1, 2), c, merge=False)
    d.add((2, 3), comp.BS(theta=1/4))
    comps.append((2, 3))

    l_comp = list(d.__iter__())

    assert len(l_comp) == 4
    for i in range(4):
        assert float(l_comp[i][1].param("theta")) == 1/(i+1) and l_comp[i][0] == comps[i]


def test_evolve():
    c = comp.BS.H()
    for backend_name in ["SLOS", "Naive"]:
        simulator = BackendFactory().get_backend(backend_name)(c)
        assert str(simulator.evolve(BasicState("|1,0>"))) == "sqrt(2)/2*|1,0>+sqrt(2)/2*|0,1>"


def _generate_simple_circuit():
    return (comp.Unitary(U=Matrix.random_unitary(3), name="U1")
            // (0, comp.PS(sp.pi / 2))
            // comp.Unitary(U=Matrix.random_unitary(3), name="U2"))


def test_visualization_ucircuit(capfd):
    c = _generate_simple_circuit()
    pdisplay(c, output_format=Format.TEXT)
    out, err = capfd.readouterr()
    assert out.strip() == """
    ╭─────╮╭───────────╮╭─────╮
0:──┤U1   ├┤PS phi=pi/2├┤U2   ├──:0 (depth 3)
    │     │╰───────────╯│     │
    │     │             │     │
1:──┤     ├─────────────┤     ├──:1 (depth 2)
    │     │             │     │
    │     │             │     │
2:──┤     ├─────────────┤     ├──:2 (depth 2)
    ╰─────╯             ╰─────╯
""".strip()


TEST_DATA_DIR = Path(__file__).resolve().parent / 'data'


def test_depths_ncomponents():
    assert comp.PS(0).depths() == [1]
    assert comp.PS(0).ncomponents() == 1
    c = _generate_simple_circuit()
    assert c.depths() == [3, 2, 2]
    assert c.ncomponents() == 3
    with open(TEST_DATA_DIR / 'u_random_8', "r") as f:
        M = Matrix(f)
        ub = (Circuit(2)
              // comp.BS()
              // (0, comp.PS(phi=P("φ_a")))
              // comp.BS()
              // (0, comp.PS(phi=P("φ_b"))))
        C1 = Circuit.decomposition(M, ub, shape="triangle")
        assert C1 is not None and C1.depths() == [28, 38, 32, 26, 20, 14, 8, 2]
        assert C1.ncomponents() == 112


def test_reflexivity():
    c = comp.BS(theta=comp.BS.r_to_theta(1/3), convention=comp.BSConvention.H)
    assert pytest.approx(c.compute_unitary(use_symbolic=False)[0, 0]) == np.sqrt(1/3)


def test_getitem1_index():
    c = Circuit(2) // comp.BS() // comp.PS(P("phi1")) // comp.BS() // comp.PS(P("phi2"))
    with pytest.raises(IndexError):
        a = c[0, 5]
    with pytest.raises(ValueError):
        a = c[-1]
    with pytest.raises(IndexError):
        a = c[4, 0]


def test_getitem2_value():
    c = Circuit(2) // comp.BS.H() // comp.PS(P("phi1")) // comp.BS.H() // comp.PS(P("phi2"))
    assert c[0, 0].describe() == "BS(convention=BSConvention.H)"
    assert c[0, 1].describe() == "PS(phi=phi1)"


def test_getitem3_parameter():
    c = Circuit(2) // comp.BS.H() // comp.PS(P("phi1")) // comp.BS.H() // comp.PS(P("phi2"))
    assert c.getitem((0, 0), True).describe() == "PS(phi=phi1)"
