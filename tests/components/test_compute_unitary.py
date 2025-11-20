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
