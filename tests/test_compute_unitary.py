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

import perceval as pcvl
import perceval.lib.symb as symb
import perceval.lib.phys as phys
import numpy as np


def _check_unitary(component: pcvl.ACircuit):
    u_symb = component.compute_unitary(use_symbolic=True)
    u_num = component.compute_unitary(use_symbolic=False)
    assert u_symb.is_unitary()
    assert u_num.is_unitary()
    assert np.allclose(u_symb.tonp(), u_num)


def test_phys_BS_unitary():
    bs = phys.BS(theta=0.43, phi_a=0.26, phi_b=1.6, phi_d=0.04)
    _check_unitary(bs)


def test_symb_BS_unitary():
    bs = symb.BS(theta=0.43, phi=0.84)
    _check_unitary(bs)


def test_phys_PS_unitary():
    bs = phys.PS(phi=1.06)
    _check_unitary(bs)


def test_symb_PS_unitary():
    bs = symb.PS(phi=0.82)
    _check_unitary(bs)


def test_phys_WP_unitary():
    wp = phys.WP(delta=0.24, xsi=0.58)
    _check_unitary(wp)


def test_symb_WP_unitary():
    wp = symb.WP(delta=0.24, xsi=0.58)
    _check_unitary(wp)


def test_phys_PR_unitary():
    wp = phys.PR(delta=0.51)
    _check_unitary(wp)


def test_symb_PR_unitary():
    wp = symb.PR(delta=0.37)
    _check_unitary(wp)
