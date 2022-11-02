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
import perceval.components.unitary_components as comp
import time

m = 16
trials = 5
# render_format = pcvl.Format.MPLOT
render_format = pcvl.Format.HTML
modes = [4, 8, 16, 24]


def generate_circuit(n_mode):
    u = pcvl.Matrix.random_unitary(n_mode)
    mzi = (pcvl.Circuit(2)
           // comp.BS()
           // (0, comp.PS(phi=pcvl.P("φ_a")))
           // comp.BS()
           // (0, comp.PS(phi=pcvl.P("φ_b"))))
    return pcvl.Circuit.decomposition(u, mzi, shape="triangle")


def benchmark_pdisplay(m, t, f):
    c = generate_circuit(m)

    render_time = 0

    for _ in range(t):
        tic = time.time()
        pcvl.pdisplay(c, output_format=f)
        # pcvl.pdisplay_to_file(c, path="tmp.svg", output_format=f)
        tac = time.time()
        render_time += tac-tic

    print("Circuit rendering benchmark results")
    print("===================================")
    print(f"Circuit containing {c.ncomponents()} components")
    print(f"with rendering method: {f.name}")
    print(f" => {render_time/trials} s (average on {trials} trials)")
    print(" ")


if __name__ == "__main__":
    for nm in modes:
        benchmark_pdisplay(nm, trials, render_format)
