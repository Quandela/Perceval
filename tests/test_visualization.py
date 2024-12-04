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

import pytest

import sys
import sympy as sp

from perceval.utils import Matrix, BasicState, Parameter, InterferometerShape
from perceval.components import (BS, PS, PBS, WP, HWP, PERM, QWP, PR, Unitary, TD, Detector, catalog, FFConfigurator,
                                 FFCircuitProvider, Circuit, GenericInterferometer, Processor)
from perceval.rendering.circuit import SymbSkin

from _test_utils import _save_or_check, save_figs


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
    _save_or_check(Circuit(4) // PERM([0, 1, 2, 3]), tmp_path, sys._getframe().f_code.co_name, save_figs)


def test_svg_dump_phys_perm4_inv(tmp_path, save_figs):
    _save_or_check(Circuit(4) // PERM([3, 2, 1, 0]), tmp_path, sys._getframe().f_code.co_name, save_figs)


def test_svg_dump_phys_perm4_swap(tmp_path, save_figs):
    _save_or_check(Circuit(4) // PERM([3, 1, 2, 0]), tmp_path, sys._getframe().f_code.co_name, save_figs)


def test_svg_dump_no_circuit_4(tmp_path, save_figs):
    _save_or_check(Circuit(4), tmp_path, sys._getframe().f_code.co_name, save_figs)


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
    nc = (Circuit(4)
          .add((0, 1), PERM([1, 0]))
          .add((1, 2), PERM([1, 0]))
          .add((2, 3), PERM([1, 0]))
          .add((1, 2), PERM([1, 0]))
          .add((0, 1), PERM([1, 0])))
    _save_or_check(nc, tmp_path, sys._getframe().f_code.co_name, save_figs)


def _create_qrng():
    chip_qrng = Circuit(4, name='QRNG')
    # Parameters
    phis = [Parameter("phi1"), Parameter("phi2"),
            Parameter("phi3"), Parameter("phi4")]
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
    ub1 = Circuit(2) // BS.H() // (0, PS(Parameter("θ"))) // BS.H() // (0, PS(Parameter("φ")))
    _save_or_check(ub1, tmp_path, sys._getframe().f_code.co_name, save_figs)


def test_svg_dump_unitary(tmp_path, save_figs):
    m = 6
    c_a = Unitary(name="W_1", U=Matrix.random_unitary(m))
    c_b = Unitary(name="W_2", U=Matrix.random_unitary(m))
    p_x = Parameter("x")
    c = (Circuit(m)
         .add(0, c_a, merge=False)
         .add(0, PS(p_x))
         .add(0, c_b, merge=False))
    _save_or_check(c, tmp_path, sys._getframe().f_code.co_name, save_figs)


def test_svg_dump_grover(tmp_path, save_figs):
    def oracle(mark):
        """Values 0, 1, 2 and 3 for parameter 'mark' respectively mark the elements "00", "01", "10" and "11" of the list."""
        oracle_circuit = Circuit(m=2, name='Oracle')
        # The following dictionary translates n into the corresponding component settings
        oracle_dict = {0: (1, 0), 1: (0, 1), 2: (1, 1), 3: (0, 0)}
        PC_state, LC_state = oracle_dict[mark]
        # Mode b
        if PC_state == 1:
            oracle_circuit.add(0, _HWP(0), merge=True)
        oracle_circuit.add(0, PR(sp.pi / 2))
        if LC_state == 1:
            oracle_circuit.add(0, _HWP(0), merge=True)
        # Mode a
        if LC_state == 1:
            oracle_circuit.add(1, _HWP(0), merge=True)
        if PC_state == 1:
            oracle_circuit.add(1, _HWP(0), merge=True)
        return oracle_circuit

    def _HWP(xsi):
        hwp = Circuit(m=1) // HWP(xsi) // PS(-sp.pi / 2)
        return hwp

    bs = BS.H(phi_tr=sp.pi / 2, phi_tl=-sp.pi / 2)
    init_circuit = Circuit(m=2, name="Initialization")
    init_circuit.add(0, _HWP(sp.pi / 8), merge=True)
    init_circuit.add((0, 1), bs)
    init_circuit.add(0, PS(-sp.pi))
    inversion_circuit = Circuit(m=2, name='Inversion')
    inversion_circuit.add((0, 1), bs)
    inversion_circuit.add(0, _HWP(sp.pi / 4), merge=True)
    inversion_circuit.add((0, 1), bs)
    detection_circuit = Circuit(m=4, name='Detection')
    detection_circuit.add((0, 1), PBS())
    detection_circuit.add((2, 3), PBS())

    grover_circuit = Circuit(m=2, name='Grover')
    grover_circuit.add((0, 1), init_circuit).add((0, 1), oracle(0)).add((0, 1), inversion_circuit)

    _save_or_check(grover_circuit, tmp_path, sys._getframe().f_code.co_name + "-rec", save_figs, recursive=True)
    _save_or_check(grover_circuit, tmp_path, sys._getframe().f_code.co_name + "-norec", save_figs, recursive=False)


def test_svg_bs_based_generic_no_phase_rectangle(tmp_path, save_figs):
    c = GenericInterferometer(5,
                                   fun_gen=lambda idx: BS.H() // PS(Parameter("phi_%d" % idx)),
                                   shape=InterferometerShape.RECTANGLE)
    _save_or_check(c, tmp_path, sys._getframe().f_code.co_name, save_figs, recursive=True)


def test_svg_bs_based_generic_with_phase_rectangle(tmp_path, save_figs):
    c = GenericInterferometer(5,
                              fun_gen=lambda idx: BS.H() // PS(Parameter("phi_%d" % idx)),
                              shape=InterferometerShape.RECTANGLE,
                              depth=10,
                              phase_shifter_fun_gen=lambda idx: PS(Parameter("theta_%d" % idx)))
    _save_or_check(c, tmp_path, sys._getframe().f_code.co_name, save_figs, recursive=True)


def test_svg_mzi_based_generic_triangle(tmp_path, save_figs):
    c = GenericInterferometer(5,
                              fun_gen=lambda idx: BS.H() // PS(Parameter("phi_%d" % idx)),
                              shape=InterferometerShape.TRIANGLE,
                              phase_shifter_fun_gen=lambda idx: PS(Parameter("theta_%d" % idx)))
    _save_or_check(c, tmp_path, sys._getframe().f_code.co_name, save_figs, recursive=True)


def test_svg_decomposition_symb_compact(tmp_path, save_figs):
    c1 = Circuit.decomposition(PERM([3, 1, 0, 2]).compute_unitary(), BS(theta=Parameter("theta")))
    _save_or_check(c1, tmp_path, sys._getframe().f_code.co_name, save_figs, recursive=True, compact=True,
                   skin_type=SymbSkin)


def test_svg_processor_with_heralds_phys(tmp_path, save_figs):
    p = catalog['klm cnot'].build_processor()
    c = Circuit(2, "Test circuit") // BS() // PS(0.3) // BS()
    pc = Processor('SLOS', c)
    pc.add_herald(1, 0)
    p.add(2, pc)
    _save_or_check(p, tmp_path, sys._getframe().f_code.co_name, save_figs, recursive=True)


def test_svg_processor_with_heralds_phys_not_recursive(tmp_path, save_figs):
    p = catalog['klm cnot'].build_processor()
    c = Circuit(2, "Test circuit") // BS() // PS(0.3) // BS()
    pc = Processor('SLOS', c)
    pc.add_herald(1, 0)
    p.add(2, pc)
    _save_or_check(p, tmp_path, sys._getframe().f_code.co_name, save_figs, recursive=False)


def test_svg_processor_with_heralds_margin_overflow_left_phys(tmp_path, save_figs):
    c = Circuit(3) // BS() // (1, BS())
    pc = Processor('SLOS', c)
    pc.add_herald(0, 0)
    _save_or_check(pc, tmp_path, sys._getframe().f_code.co_name, save_figs, recursive=True)


def test_svg_processor_with_heralds_margin_overflow_right_phys(tmp_path, save_figs):
    c = Circuit(4) // (1, BS()) //  BS()
    pc = Processor('SLOS', c)
    pc.add_herald(0, 0)
    pc.add_herald(2, 1)
    _save_or_check(pc, tmp_path, sys._getframe().f_code.co_name, save_figs, recursive=True)


def test_svg_processor_with_heralds_margin_overflow_left_right_phys(tmp_path, save_figs):
    c = Circuit(2) // BS()
    pc = Processor('SLOS', c)
    pc.add_herald(0, 0)
    _save_or_check(pc, tmp_path, sys._getframe().f_code.co_name, save_figs, recursive=True)


def test_svg_processor_with_heralds_perm_following_phys(tmp_path, save_figs):
    c = Circuit(4) // (1, PERM([1, 0])) // (1, BS()) // (0, PERM([1, 0])) // BS() // (1, PERM([1, 0]))
    pc = Processor('SLOS', c)
    pc.add_herald(0, 0)
    pc.add_herald(2, 1)
    _save_or_check(pc, tmp_path, sys._getframe().f_code.co_name, save_figs, recursive=True)


def test_svg_processor_with_heralds_and_barriers_phys(tmp_path, save_figs):
    c = Circuit(4) // (1, PERM([1, 0])) @ (1, BS()) // (0, PERM([1, 0])) // BS() @ (1, PERM([1, 0]))
    pc = Processor('SLOS', c)
    pc.add_herald(0, 0)
    pc.add_herald(2, 1)
    pc.add(0, Detector.pnr())
    pc.add(1, Detector.ppnr(0.5))
    pc.add(2, Detector.threshold())
    _save_or_check(pc, tmp_path, sys._getframe().f_code.co_name, save_figs, recursive=True)


def test_svg_dump_barrier_phys(tmp_path, save_figs):
    c = Circuit(4) // BS() @ (2, BS()) // (1, BS()) @ BS()
    _save_or_check(c, tmp_path, sys._getframe().f_code.co_name, save_figs, recursive=True)


def test_svg_dump_barrier_symb(tmp_path, save_figs):
    c = Circuit(4) // BS() @ (2, BS()) // (1, BS()) @ BS()
    _save_or_check(c, tmp_path, sys._getframe().f_code.co_name, save_figs, recursive=True, skin_type=SymbSkin)


@pytest.mark.parametrize("merge_pre_MZI", [False, True])
@pytest.mark.parametrize("merge_upper_MZI", [False, True])
@pytest.mark.parametrize("merge_lower_MZI", [False, True])
def test_svg_dump_circuit_box_bell_state(tmp_path, save_figs,
                                         merge_pre_MZI,
                                         merge_upper_MZI,
                                         merge_lower_MZI):
    pre_MZI = (Circuit(4, name="Bell State Prep")
               .add(0, BS())
               .add(2, BS())
               .add(1, PERM([1, 0])))

    upper_MZI = (Circuit(2, name="upper MZI")
                 .add(0, PS(phi=Parameter('phi_0')))
                 .add(0, BS())
                 .add(0, PS(phi=Parameter('phi_2')))
                 .add(0, BS()))

    lower_MZI = (Circuit(2, name="lower MZI")
                 .add(0, PS(phi=Parameter('phi_1')))
                 .add(0, BS())
                 .add(0, PS(phi=Parameter('phi_3')))
                 .add(0, BS()))

    chip = (Circuit(4)
            .add(0, pre_MZI, merge=merge_pre_MZI)
            .add(0, upper_MZI, merge=merge_upper_MZI)
            .add(2, lower_MZI, merge=merge_lower_MZI))

    processor = Processor('SLOS', chip)

    fig_name = sys._getframe().f_code.co_name
    fig_name += "T" if merge_pre_MZI else "F"
    fig_name += "T" if merge_upper_MZI else "F"
    fig_name += "T" if merge_lower_MZI else "F"
    _save_or_check(c=processor, tmp_path=tmp_path, circuit_name=fig_name, save_figs=save_figs, recursive=True)


def create_processor_with_feed_forward():
    u = Unitary(Matrix.random_unitary(4), "U0")
    p = Processor("SLOS", u)

    mzi = catalog["mzi phase last"].build_circuit()
    mzi.name = "U1"
    ffc1 = FFConfigurator(1, -1, mzi, {"phi_a": 0, "phi_b": 0}, name="D1")

    p.add(2, Detector.pnr())
    p.add(2, ffc1)

    p.add(0, Detector.pnr())
    p.add(1, Detector.pnr())

    ffc2 = FFCircuitProvider(2, 1, Unitary(Matrix.random_unitary(1)), name="D2")
    p.add(0, ffc2)

    p.with_input(BasicState([1, 0, 1, 0]))
    return p


def test_svg_dump_feed_forward_symb(tmp_path, save_figs):
    p = create_processor_with_feed_forward()
    _save_or_check(p, tmp_path, sys._getframe().f_code.co_name, save_figs, recursive=True, skin_type=SymbSkin)


def test_svg_dump_feed_forward_phys(tmp_path, save_figs):
    p = create_processor_with_feed_forward()
    _save_or_check(p, tmp_path, sys._getframe().f_code.co_name, save_figs, recursive=True)

def test_svg_dump_feed_forward_phys_norec(tmp_path, save_figs):
    p = create_processor_with_feed_forward()
    _save_or_check(p, tmp_path, sys._getframe().f_code.co_name, save_figs, recursive=False)
