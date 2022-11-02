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

import numpy as np

from perceval import Circuit, P, Matrix
import perceval.components.unitary_components as comp

TEST_DATA_DIR = Path(__file__).resolve().parent / 'data'


def test_basic_transfer():
    a = comp.BS()
    theta = P("theta")
    b = comp.BS(theta=theta)
    b.transfer_from(a)
    assert theta.defined and pytest.approx(np.pi/2) == float(theta)


def test_basic_transfer_fix():
    a = comp.BS(phi_tr=0.1)
    theta = P("theta")
    b = comp.BS(theta=theta)
    b.transfer_from(a)
    assert pytest.approx(3*np.pi/2) != float(b.param("phi_tr"))
    b.transfer_from(a, force=True)
    assert pytest.approx(0.1) == float(b.param("phi_tr"))


def test_transfer_complex_1():
    a = Circuit(3) // (0, comp.PS(0)) // (2, comp.PS(np.pi/2)) // (0, comp.BS())
    phi_a = P("phi_a")
    phi_b = P("phi_b")
    theta = P("theta")
    b = Circuit(3) // (0, comp.PS(phi_a)) // (2, comp.PS(phi_b)) // (0, comp.BS(theta=theta))
    b.transfer_from(a)
    assert pytest.approx(0) == float(phi_a)
    assert pytest.approx(np.pi/2) == float(phi_b)
    assert pytest.approx(np.pi/2) == float(theta)


def test_transfer_complex_2():
    # order is not important
    a = Circuit(3) // (0, comp.PS(0)) // (2, comp.PS(np.pi/2)) // (0, comp.BS())
    phi_a = P("phi_a")
    phi_b = P("phi_b")
    theta = P("theta")
    b = Circuit(3) // (2, comp.PS(phi_b)) // (0, comp.PS(phi_a)) // (0, comp.BS(theta=theta))
    b.transfer_from(a)
    assert pytest.approx(0) == float(phi_a)
    assert pytest.approx(np.pi/2) == float(phi_b)
    assert pytest.approx(np.pi/2) == float(theta)


def test_transfer_complex_3():
    # the circuit can be bigger
    a = Circuit(3) // (0, comp.PS(0)) // (2, comp.PS(np.pi/2)) // (0, comp.BS()) // (1, comp.PS(0))
    phi_a = P("phi_a")
    phi_b = P("phi_b")
    theta = P("theta")
    b = Circuit(3) // (2, comp.PS(phi_b)) // (0, comp.PS(phi_a)) // (0, comp.BS(theta=theta))
    b.transfer_from(a)
    assert pytest.approx(0) == float(phi_a)
    assert pytest.approx(np.pi/2) == float(phi_b)
    assert pytest.approx(np.pi/2) == float(theta)


def test_transfer_complex_4():
    # but the circuit cannot match a component that is not here
    a = Circuit(3) // (0, comp.PS(0)) // (2, comp.PS(np.pi/2)) // (0, comp.BS()) // (1, comp.PS(0))
    phi_a = P("phi_a")
    phi_b = P("phi_b")
    phi_c = P("phi_c")
    theta = P("theta")
    b = Circuit(3) // (2, comp.PS(phi_b)) // (1, comp.PS(phi_c)) // (0, comp.PS(phi_a)) // (0, comp.BS(theta=theta))
    with pytest.raises(AssertionError):
        b.transfer_from(a)
    a = Circuit(3) // (0, comp.PS(0)) // (1, comp.PS(0)) // (2, comp.PS(np.pi/2)) // (0, comp.BS())
    phi_a = P("phi_a")
    phi_b = P("phi_b")
    phi_c = P("phi_c")
    theta = P("theta")
    b = Circuit(3) // (2, comp.PS(phi_b)) // (0, comp.PS(phi_a)) // (0, comp.BS(theta=theta)) // (1, comp.PS(phi_c))
    with pytest.raises(AssertionError):
        b.transfer_from(a)


def test_transfer_complex_5():
    # check a generic decomposition
    with open(TEST_DATA_DIR / 'u_random_8', "r") as f:
        M = Matrix(f)
        def ub(idx):
            return (Circuit(2)
                    // comp.BS()
                    // (0, comp.PS(phi=P("φ_a%d" % idx)))
                    // comp.BS()
                    // (0, comp.PS(phi=P("φ_b%d" % idx))))
        C1 = Circuit.decomposition(M, ub(0), shape="triangle")
    C2 = Circuit.generic_interferometer(8, ub, shape="triangle")
    C2.transfer_from(C1)
    def ub_varbs(idx):
        return (Circuit(2)
                // comp.BS(theta=P("theta%d" % (2*idx)))
                // (0, comp.PS(phi=0))
                // comp.BS(theta=P("theta%d" % (2*idx+1)))
                // (0, comp.PS(phi=0)))
    C3 = Circuit.generic_interferometer(8, ub_varbs, shape="triangle")
    C3.transfer_from(C1, force=True)
