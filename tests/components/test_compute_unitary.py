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

import perceval as pcvl
import perceval.components.unitary_components as comp
import numpy as np


def _check_unitary(component: pcvl.ACircuit):
    u_symb = component.compute_unitary(use_symbolic=True)
    u_num = component.compute_unitary(use_symbolic=False)
    assert u_symb.is_unitary()
    assert u_num.is_unitary()
    assert np.allclose(u_symb.tonp(), u_num)


def test_BS_unitary():
    bs = comp.BS(theta=0.43, phi_tl=0.26, phi_bl=1.6, phi_tr=0.04, phi_br=2.13)
    _check_unitary(bs)
    bs = comp.BS.H(theta=0.43, phi_tl=0.26, phi_bl=1.6, phi_tr=0.04, phi_br=2.13)
    _check_unitary(bs)
    bs = comp.BS.Ry(theta=0.43, phi_tl=0.26, phi_bl=1.6, phi_tr=0.04, phi_br=2.13)
    _check_unitary(bs)


def test_PS_unitary():
    bs = comp.PS(phi=0.82)
    _check_unitary(bs)


def test_WP_unitary():
    wp = comp.WP(delta=0.24, xsi=0.58)
    _check_unitary(wp)


def test_PR_unitary():
    wp = comp.PR(delta=0.37)
    _check_unitary(wp)


def _reference_unitary(circuit: pcvl.Circuit, use_polarization: bool = False):
    # Explicit product of identity-padded component matrices
    multiplier = 2 if use_polarization else 1
    u = np.eye(multiplier * circuit.m, dtype=complex)
    for r, c in circuit._components:
        padded = np.eye(multiplier * circuit.m, dtype=complex)
        start, end = multiplier * r[0], multiplier * (r[-1] + 1)
        padded[start:end, start:end] = c.compute_unitary(use_polarization=use_polarization)
        u = padded @ u
    return u


def test_circuit_unitary_matches_padded_product():
    u_sub = pcvl.Matrix.random_unitary(3)
    sub = pcvl.Circuit(3) // comp.BS(0.7) // (1, comp.PS(0.3)) // (1, comp.BS.H(1.2))
    c = (pcvl.Circuit(5) // (1, comp.BS(0.2)) // (4, comp.PS(1.1)) // (2, comp.Unitary(u_sub))
         // (0, comp.BS.Rx(0.5, phi_tl=0.1)) // (3, comp.PERM([1, 0])))
    c.add(1, sub, merge=False)
    c.add(0, comp.Unitary(pcvl.Matrix.random_unitary(5)), merge=False)
    u = c.compute_unitary(use_symbolic=False)
    assert isinstance(u, pcvl.MatrixN)
    assert u.is_unitary()
    assert np.allclose(u, _reference_unitary(c), atol=1e-12, rtol=1e-12)
    assert np.allclose(u, c.compute_unitary(use_symbolic=True).tonp())

    c_pol = pcvl.Circuit(3) // (0, comp.BS(0.3)) // (1, comp.WP(0.2, 0.7)) // (2, comp.PR(0.4)) // (1, comp.PBS())
    u_pol = c_pol.compute_unitary(use_polarization=True)
    assert u_pol.shape == (6, 6)
    assert np.allclose(u_pol, _reference_unitary(c_pol, use_polarization=True), atol=1e-12, rtol=1e-12)


def test_circuit_unitary_does_not_alias_component():
    u_ref = pcvl.Matrix.random_unitary(3)
    unitary = comp.Unitary(u_ref)
    c = pcvl.Circuit(3) // unitary
    u = c.compute_unitary()
    assert np.allclose(u, u_ref)
    u[0, 0] = 42
    assert np.allclose(unitary.compute_unitary(), u_ref)
    assert np.allclose(pcvl.Circuit(2).compute_unitary(), np.eye(2))
