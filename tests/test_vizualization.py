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
from typing import Type

import pytest

import perceval as pcvl
from perceval.components.unitary_components import *
from perceval.components.non_unitary_components import *
from perceval.rendering import pdisplay_to_file, Format
from pathlib import Path
import re
import sympy as sp
from perceval.rendering.circuit import ASkin, SymbSkin, PhysSkin

TEST_IMG_DIR = Path(__file__).resolve().parent / 'imgs'


@pytest.fixture(scope="session")
def save_figs(pytestconfig):
    return pytestconfig.getoption("save_figs")


def _norm(svg):
    svg = svg.replace(" \n", "\n")
    svg = re.sub(r'url\(#.*?\)', 'url(#staticClipPath)', svg)
    svg = re.sub(r'<clipPath id=".*?">', '<clipPath id="staticClipPath">', svg)
    svg = re.sub(r'<dc:date>(.*)</dc:date>', '<dc:date></dc:date>', svg)
    svg = re.sub(r'<dc:title>(.*)</dc:title>', '<dc:title></dc:title>', svg)
    return svg


def _check_image(test_path, ref_path):
    with open(test_path) as f_test:
        test = _norm("".join(f_test.readlines()))
    with open(ref_path) as f_ref:
        ref = _norm("".join(f_ref.readlines()))
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


def _save_or_check(c, tmp_path, circuit_name, save_figs, recursive=False, compact=False,
                   skin_type: Type[ASkin] = PhysSkin) -> None:
    img_path = (TEST_IMG_DIR if save_figs else tmp_path) / Path(circuit_name + ".svg")
    skin = skin_type(compact)
    pdisplay_to_file(c, img_path, output_format=Format.MPLOT, recursive=recursive, skin=skin)

    if save_figs:
        with open(img_path) as f_saved:
            saved = "".join(f_saved.readlines())
        saved = _norm(saved)
        with open(img_path, "w") as fw_saved:
            fw_saved.write(saved)
    else:
        ok, msg = _check_image(img_path, TEST_IMG_DIR / Path(circuit_name + ".svg"))
        assert ok, msg


def test_svg_dump_phys_bs(tmp_path, save_figs):
    _save_or_check(BS.H(), tmp_path, sys._getframe().f_code.co_name, save_figs)


def test_svg_dump_phys_ps(tmp_path, save_figs):
    _save_or_check(PS(sp.pi / 2), tmp_path, sys._getframe().f_code.co_name, save_figs)


def test_svg_dump_phys_pbs(tmp_path, save_figs):
    _save_or_check(PBS(), tmp_path, sys._getframe().f_code.co_name, save_figs)


def test_svg_dump_phys_dt(tmp_path, save_figs):
    _save_or_check(TD(0), tmp_path, sys._getframe().f_code.co_name, save_figs)


def test_svg_dump_phys_wp(tmp_path, save_figs):
    _save_or_check(WP(sp.pi / 4, sp.pi / 4), tmp_path, sys._getframe().f_code.co_name, save_figs)


def test_svg_dump_phys_hwp(tmp_path, save_figs):
    _save_or_check(HWP(sp.pi / 2), tmp_path, sys._getframe().f_code.co_name, save_figs)


def test_svg_dump_phys_qwp(tmp_path, save_figs):
    _save_or_check(QWP(sp.pi / 4), tmp_path, sys._getframe().f_code.co_name, save_figs)


def test_svg_dump_phys_pr(tmp_path, save_figs):
    _save_or_check(PR(sp.pi / 4), tmp_path, sys._getframe().f_code.co_name, save_figs)


def test_svg_dump_phys_perm4_0(tmp_path, save_figs):
    _save_or_check(pcvl.Circuit(4) // PERM([0, 1, 2, 3]), tmp_path, sys._getframe().f_code.co_name, save_figs)


def test_svg_dump_phys_perm4_inv(tmp_path, save_figs):
    _save_or_check(pcvl.Circuit(4) // PERM([3, 2, 1, 0]), tmp_path, sys._getframe().f_code.co_name, save_figs)


def test_svg_dump_phys_perm4_swap(tmp_path, save_figs):
    _save_or_check(pcvl.Circuit(4) // PERM([3, 1, 2, 0]), tmp_path, sys._getframe().f_code.co_name, save_figs)


def test_svg_dump_no_circuit_4(tmp_path, save_figs):
    _save_or_check(pcvl.Circuit(4), tmp_path, sys._getframe().f_code.co_name, save_figs)


def test_svg_dump_symb_bs_compact(tmp_path, save_figs):
    _save_or_check(BS(BS.r_to_theta(1/3)), tmp_path, sys._getframe().f_code.co_name, save_figs, compact=True,
                   skin_type=SymbSkin)


def test_svg_dump_symb_bs_compact_false(tmp_path, save_figs):
    _save_or_check(BS(BS.r_to_theta(1/3)), tmp_path, sys._getframe().f_code.co_name, save_figs, compact=False,
                   skin_type=SymbSkin)


def test_svg_dump_symb_ps(tmp_path, save_figs):
    _save_or_check(PS(sp.pi / 2), tmp_path, sys._getframe().f_code.co_name, save_figs, skin_type=SymbSkin)


def test_svg_dump_symb_pbs_compact(tmp_path, save_figs):
    _save_or_check(PBS(), tmp_path, sys._getframe().f_code.co_name, save_figs, compact=True, skin_type=SymbSkin)


def test_svg_dump_symb_pbs_compact_false(tmp_path, save_figs):
    _save_or_check(PBS(), tmp_path, sys._getframe().f_code.co_name, save_figs, compact=False, skin_type=SymbSkin)


def test_svg_dump_symb_pr(tmp_path, save_figs):
    _save_or_check(PR(sp.pi / 4), tmp_path, sys._getframe().f_code.co_name, save_figs, skin_type=SymbSkin)


def test_svg_dump_symb_wp(tmp_path, save_figs):
    _save_or_check(WP(sp.pi / 4, sp.pi / 4), tmp_path, sys._getframe().f_code.co_name, save_figs, skin_type=SymbSkin)


def test_svg_dump_symb_hwp(tmp_path, save_figs):
    _save_or_check(HWP(sp.pi / 2), tmp_path, sys._getframe().f_code.co_name, save_figs, skin_type=SymbSkin)


def test_svg_dump_symb_qwp(tmp_path, save_figs):
    _save_or_check(QWP(sp.pi / 4), tmp_path, sys._getframe().f_code.co_name, save_figs, skin_type=SymbSkin)


def test_svg_dump_phys_multi_perm(tmp_path, save_figs):
    nc = (pcvl.Circuit(4)
          .add((0, 1), PERM([1, 0]))
          .add((1, 2), PERM([1, 0]))
          .add((2, 3), PERM([1, 0]))
          .add((1, 2), PERM([1, 0]))
          .add((0, 1), PERM([1, 0])))
    _save_or_check(nc, tmp_path, sys._getframe().f_code.co_name, save_figs)


def _create_qrng():
    chip_qrng = pcvl.Circuit(4, name='QRNG')
    # Parameters
    phis = [pcvl.Parameter("phi1"), pcvl.Parameter("phi2"),
            pcvl.Parameter("phi3"), pcvl.Parameter("phi4")]
    return (chip_qrng
            .add((0, 1), BS())
            .add((2, 3), BS())
            .add((1, 2), PERM([1, 0]))
            .add(0, PS(phis[0]))
            .add(2, PS(phis[2]))
            .add((0, 1), BS())
            .add((2, 3), BS())
            .add(0, PS(phis[1]))
            .add(2, PS(phis[3]))
            .add((0, 1), BS())
            .add((2, 3), BS())
            )


def test_svg_dump_qrng(tmp_path, save_figs):
    c = _create_qrng()
    _save_or_check(c, tmp_path, sys._getframe().f_code.co_name, save_figs, compact=False, skin_type=SymbSkin)


def test_svg_dump_qrng_compact(tmp_path, save_figs):
    c = _create_qrng()
    _save_or_check(c, tmp_path, sys._getframe().f_code.co_name, save_figs, compact=True, skin_type=SymbSkin)


def test_svg_dump_phys_universal1(tmp_path, save_figs):
    ub1 = pcvl.Circuit(2) // BS.H() // (0, PS(pcvl.P("θ"))) // BS.H() // (0, PS(pcvl.P("φ")))
    _save_or_check(ub1, tmp_path, sys._getframe().f_code.co_name, save_figs)


def test_svg_dump_unitary(tmp_path, save_figs):
    m = 6
    c_a = Unitary(name="W_1", U=pcvl.Matrix.random_unitary(m))
    c_b = Unitary(name="W_2", U=pcvl.Matrix.random_unitary(m))
    p_x = pcvl.P("x")
    c = (pcvl.Circuit(m)
         .add(0, c_a, merge=False)
         .add(0, PS(p_x))
         .add(0, c_b, merge=False))
    _save_or_check(c, tmp_path, sys._getframe().f_code.co_name, save_figs)


def test_svg_dump_grover(tmp_path, save_figs):
    def oracle(mark):
        """Values 0, 1, 2 and 3 for parameter 'mark' respectively mark the elements "00", "01", "10" and "11" of the list."""
        oracle_circuit = pcvl.Circuit(m=2, name='Oracle')
        # The following dictionnary translates n into the corresponding component settings
        oracle_dict = {0: (1, 0), 1: (0, 1), 2: (1, 1), 3: (0, 0)}
        PC_state, LC_state = oracle_dict[mark]
        # Mode b
        if PC_state == 1:
            oracle_circuit.add(0, _HWP(0))
        oracle_circuit.add(0, PR(sp.pi / 2))
        if LC_state == 1:
            oracle_circuit.add(0, _HWP(0))
        # Mode a
        if LC_state == 1:
            oracle_circuit.add(1, _HWP(0))
        if PC_state == 1:
            oracle_circuit.add(1, _HWP(0))
        return oracle_circuit

    def _HWP(xsi):
        hwp = pcvl.Circuit(m=1)
        hwp.add(0, HWP(xsi)).add(0, PS(-sp.pi / 2))
        return hwp

    bs = BS.H(phi_tr=sp.pi / 2, phi_tl=-sp.pi / 2)
    init_circuit = pcvl.Circuit(m=2, name="Initialization")
    init_circuit.add(0, _HWP(sp.pi / 8))
    init_circuit.add((0, 1), bs)
    init_circuit.add(0, PS(-sp.pi))
    inversion_circuit = pcvl.Circuit(m=2, name='Inversion')
    inversion_circuit.add((0, 1), bs)
    inversion_circuit.add(0, _HWP(sp.pi / 4))
    inversion_circuit.add((0, 1), bs)
    detection_circuit = pcvl.Circuit(m=4, name='Detection')
    detection_circuit.add((0, 1), PBS())
    detection_circuit.add((2, 3), PBS())

    grover_circuit = pcvl.Circuit(m=2, name='Grover')
    grover_circuit.add((0, 1), init_circuit).add((0, 1), oracle(0)).add((0, 1), inversion_circuit)

    _save_or_check(grover_circuit, tmp_path, sys._getframe().f_code.co_name + "-rec", save_figs, recursive=True)
    _save_or_check(grover_circuit, tmp_path, sys._getframe().f_code.co_name + "-norec", save_figs, recursive=False)


def test_svg_bs_based_generic_no_phase_rectangle(tmp_path, save_figs):
    c = pcvl.Circuit.generic_interferometer(5,
                                            fun_gen=lambda idx: BS.H() // PS(pcvl.P("φ_%d" % idx)),
                                            shape="rectangle")
    _save_or_check(c, tmp_path, sys._getframe().f_code.co_name, save_figs, recursive=True)


def test_svg_bs_based_generic_with_phase_rectangle(tmp_path, save_figs):
    c = pcvl.Circuit.generic_interferometer(5,
                                            fun_gen=lambda idx: BS.H() // PS(pcvl.P("φ_%d" % idx)),
                                            shape="rectangle",
                                            depth=10,
                                            phase_shifter_fun_gen=lambda idx: PS(pcvl.P("Φ_%d" % idx)))
    _save_or_check(c, tmp_path, sys._getframe().f_code.co_name, save_figs, recursive=True)


def test_svg_mzi_based_generic_triangle(tmp_path, save_figs):
    c = pcvl.Circuit.generic_interferometer(5,
                                            fun_gen=lambda idx: BS.H() // PS(pcvl.P("φ_%d" % idx)),
                                            shape="triangle",
                                            phase_shifter_fun_gen=lambda idx: PS(pcvl.P("Φ_%d" % idx)))
    _save_or_check(c, tmp_path, sys._getframe().f_code.co_name, save_figs, recursive=True)


def test_svg_decomposition_symb_compact(tmp_path, save_figs):
    c1 = pcvl.Circuit.decomposition(pcvl.Matrix(PERM([3, 1, 0, 2]).U), BS(theta=pcvl.P("theta")))
    _save_or_check(c1, tmp_path, sys._getframe().f_code.co_name, save_figs, recursive=True, compact=True,
                   skin_type=SymbSkin)


def test_svg_processor_with_heralds_phys(tmp_path, save_figs):
    p = pcvl.components.catalog['heralded cnot'].build()
    c = pcvl.Circuit(2, "Test circuit") // BS() // PS(0.3) // BS()
    pc = pcvl.Processor('SLOS', c)
    pc.add_herald(1, 0)
    p.add(2, pc)
    _save_or_check(p, tmp_path, sys._getframe().f_code.co_name, save_figs, recursive=True)
