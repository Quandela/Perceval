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
import perceval.components.unitary_components as comp
import perceval.algorithm as algo
from perceval.rendering.pdisplay import pdisplay_analyzer
import sympy as sp

from test_circuit import strip_line_12


def test_analyser_on_qrng():
    chip_QRNG = pcvl.Circuit(4, name='QRNG')
    # Parameters
    phis = [pcvl.Parameter("phi1"), pcvl.Parameter("phi2"),
            pcvl.Parameter("phi3"), pcvl.Parameter("phi4")]
    # Defining the LO elements of chip
    (chip_QRNG
     .add((0, 1), comp.BS())
     .add((2, 3), comp.BS())
     .add((1, 2), comp.PERM([1, 0]))
     .add(0, comp.PS(phis[0]))
     .add(2, comp.PS(phis[2]))
     .add((0, 1), comp.BS())
     .add((2, 3), comp.BS())
     .add(0, comp.PS(phis[1]))
     .add(2, comp.PS(phis[3]))
     .add((0, 1), comp.BS())
     .add((2, 3), comp.BS())
     )
    # Setting parameters value and see how chip specs evolve
    phis[0].set_value(sp.pi/2)
    phis[1].set_value(0.2)
    phis[2].set_value(0)
    phis[3].set_value(0.4)

    p = pcvl.Processor("Naive", chip_QRNG)

    ca = algo.Analyzer(p, [pcvl.BasicState("[1,0,1,0]"), pcvl.BasicState("[0,1,1,0]")], "*")
    ca.compute()
    assert strip_line_12(pdisplay_analyzer(ca)) == strip_line_12("""
            +-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
            |           | |1,0,1,0> | |1,1,0,0> | |0,2,0,0> | |2,0,0,0> | |1,0,0,1> | |0,1,1,0> | |0,1,0,1> | |0,0,2,0> | |0,0,1,1> | |0,0,0,2> |
            +-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
            | |1,0,1,0> | 0.012162  | 0.240133  | 0.004934  | 0.004934  | 0.237838  | 0.237838  | 0.012162  | 0.018956  | 0.212088  | 0.018956  |
            | |0,1,1,0> | 0.012162  | 0.240133  | 0.004934  | 0.004934  | 0.237838  | 0.237838  | 0.012162  | 0.018956  | 0.212088  | 0.018956  |
            +-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
        """).strip()
