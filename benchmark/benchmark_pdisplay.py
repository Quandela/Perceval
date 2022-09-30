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

r"""
This script times the execution of pdisplay for circuits.
"""

import perceval as pcvl
import perceval.lib.phys as phys

trials = 2


def generate_circuit(n_mode):
    u = pcvl.Matrix.random_unitary(n_mode)
    mzi = (pcvl.Circuit(2)
           // phys.BS()
           // (0, phys.PS(phi=pcvl.P("φ_a")))
           // phys.BS()
           // (0, phys.PS(phi=pcvl.P("φ_b"))))
    return pcvl.Circuit.decomposition(u, mzi, shape="triangle")


c6 = generate_circuit(6)
c12 = generate_circuit(12)


def run_pdisplay(c, t, f):
    for _ in range(t):
        pcvl.pdisplay(c, output_format=f, mplot_noshow=True)


def _run_pdisplay_mplot_6():
    run_pdisplay(c6, trials, "mplot")


def _run_pdisplay_mplot_12():
    run_pdisplay(c12, trials, "mplot")


def _run_pdisplay_svg_6():
    run_pdisplay(c6, trials, "html")


def _run_pdisplay_svg_12():
    run_pdisplay(c12, trials, "html")


def test_pdisplay_mplot_6(benchmark):
    benchmark(_run_pdisplay_mplot_6)


def test_pdisplay_mplot_12(benchmark):
    benchmark(_run_pdisplay_mplot_12)


def test_pdisplay_svg_6(benchmark):
    benchmark(_run_pdisplay_svg_6)


def test_pdisplay_svg_12(benchmark):
    benchmark(_run_pdisplay_svg_12)
