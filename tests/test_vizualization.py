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

import sys
import pytest

import perceval as pcvl
import perceval.lib.phys as phys
import perceval.lib.symb as symb
from pathlib import Path
import re
import sympy as sp

TEST_IMG_DIR = Path(__file__).resolve().parent / 'imgs'

@pytest.fixture(scope="session")
def save_figs(pytestconfig):
    return pytestconfig.getoption("save_figs")

def _check_image(test_path, ref_path):
    with open(test_path) as f_test:
        test = "".join(f_test.readlines()).replace(" \n", "\n")
    with open(ref_path) as f_ref:
        ref = "".join(f_ref.readlines()).replace(" \n", "\n")
    m_test = re.search(r'<g id="PatchCollection.*?>((.|\n)*?)</g>', test)
    m_ref = re.search(r'<g id="PatchCollection.*?>((.|\n)*?)</g>', ref)
    if not m_test:
        return False, "cannot find patch in test"
    if not m_ref:
        return False, "cannot find patch in ref"
    m_test_patch = re.sub(r'url\(#.*?\)', "url()", m_test.group(1))
    m_ref_patch = re.sub(r'url\(#.*?\)', "url()", m_ref.group(1))
    if m_test_patch != m_ref_patch:
        return False, "test and ref are different"
    return True, "ok"


def _save_or_check(c, tmp_path, circuit_name, save_figs):
    if save_figs:
        c.pdisplay(output_format="mplot",
                   mplot_savefig=TEST_IMG_DIR / Path(circuit_name + ".svg"),
                   mplot_noshow=True)
    else:
        c.pdisplay(output_format="mplot",
                   mplot_savefig=tmp_path / Path(circuit_name + ".svg"),
                   mplot_noshow=True)
        ok, msg = _check_image(tmp_path / Path(circuit_name + ".svg"),
                               TEST_IMG_DIR / Path(circuit_name + ".svg"))
        assert ok, msg


def test_svg_dump_phys_bs(tmp_path, save_figs):
    _save_or_check(phys.BS(), tmp_path, sys._getframe().f_code.co_name, save_figs)


def test_svg_dump_phys_ps(tmp_path, save_figs):
    _save_or_check(phys.PS(sp.pi/2), tmp_path, sys._getframe().f_code.co_name, save_figs)


def test_svg_dump_phys_pbs(tmp_path, save_figs):
    _save_or_check(phys.PBS(), tmp_path, sys._getframe().f_code.co_name, save_figs)


def test_svg_dump_phys_dt(tmp_path, save_figs):
    _save_or_check(phys.DT(0), tmp_path, sys._getframe().f_code.co_name, save_figs)


def test_svg_dump_phys_wp(tmp_path, save_figs):
    _save_or_check(phys.WP(sp.pi/4, sp.pi/4), tmp_path, sys._getframe().f_code.co_name, save_figs)


def test_svg_dump_phys_pr(tmp_path, save_figs):
    _save_or_check(phys.PR(sp.pi/4), tmp_path, sys._getframe().f_code.co_name, save_figs)


def test_svg_dump_phys_perm4_0(tmp_path, save_figs):
    _save_or_check(pcvl.Circuit(4) // phys.PERM([0, 1, 2, 3]), tmp_path, sys._getframe().f_code.co_name, save_figs)


def test_svg_dump_phys_perm4_inv(tmp_path, save_figs):
    _save_or_check(pcvl.Circuit(4) // phys.PERM([3, 2, 1, 0]), tmp_path, sys._getframe().f_code.co_name, save_figs)


def test_svg_dump_phys_perm4_swap(tmp_path, save_figs):
    _save_or_check(pcvl.Circuit(4) // phys.PERM([3, 1, 2, 0]), tmp_path, sys._getframe().f_code.co_name, save_figs)


def test_svg_dump_no_circuit_4(tmp_path, save_figs):
    _save_or_check(pcvl.Circuit(4), tmp_path, sys._getframe().f_code.co_name, save_figs)


def test_svg_dump_qrng(tmp_path, save_figs):
    chip_QRNG = pcvl.Circuit(4, name='QRNG')
    # Parameters
    phis = [pcvl.Parameter("phi1"), pcvl.Parameter("phi2"),
            pcvl.Parameter("phi3"), pcvl.Parameter("phi4")]
    c = (chip_QRNG
             .add((0, 1), symb.BS())
             .add((2, 3), symb.BS())
             .add((1, 2), symb.PERM([1, 0]))
             .add(0, symb.PS(phis[0]))
             .add(2, symb.PS(phis[2]))
             .add((0, 1), symb.BS())
             .add((2, 3), symb.BS())
             .add(0, symb.PS(phis[1]))
             .add(2, symb.PS(phis[3]))
             .add((0, 1), symb.BS())
             .add((2, 3), symb.BS())
    )
    _save_or_check(c, tmp_path, sys._getframe().f_code.co_name, save_figs)
