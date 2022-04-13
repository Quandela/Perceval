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

from pytest import approx
import perceval as pcvl
import perceval.lib.symb as symb


def test_perm():
    c = pcvl.Circuit(4).add(0, symb.PERM([3, 1, 2, 0]))
    ub = (pcvl.Circuit(2)
          // (0, symb.PS(phi=pcvl.Parameter("φ_a")))
          // symb.BS()
          // (0, symb.PS(phi=pcvl.Parameter("φ_b")))
          // symb.BS())
    C1 = pcvl.Circuit.decomposition(pcvl.Matrix(c.U), ub, phase_shifter_fn=symb.PS, shape="triangle",
                                    permutation=symb.PERM)
    assert C1.describe().replace("\n", "").replace(" ", "") == """
        Circuit(4).add([0, 1], symb.PERM([1, 0]))
                  .add([1, 2], symb.PERM([1, 0]))
                  .add([2, 3], symb.PERM([1, 0]))
                  .add([1, 2], symb.PERM([1, 0]))
                  .add([0, 1], symb.PERM([1, 0]))""".replace("\n", "").replace(" ", "")


def test_basic_perm_triangle():
    c = pcvl.Circuit(2).add(0, symb.PERM([1,0]))
    ub = (pcvl.Circuit(2)
          // (0, symb.PS(phi=pcvl.Parameter("φ_a")))
          // symb.BS()
          // (0, symb.PS(phi=pcvl.Parameter("φ_b")))
          // symb.BS())
    C1 = pcvl.Circuit.decomposition(pcvl.Matrix(c.U), ub, shape="triangle")
    M1 = C1.compute_unitary(use_symbolic=False)
    assert approx(1) == abs(M1[0][0])+1
    assert approx(1) == abs(M1[0][1])
    assert approx(1) == abs(M1[1][0])
    assert approx(1) == abs(M1[1][1])+1


def test_basic_perm_rectangle():
    c = pcvl.Circuit(2).add(0, symb.PERM([1,0]))
    ub = (pcvl.Circuit(2)
          // (0, symb.PS(phi=pcvl.Parameter("φ_a")))
          // symb.BS()
          // (0, symb.PS(phi=pcvl.Parameter("φ_b")))
          // symb.BS())
    C1 = pcvl.Circuit.decomposition(pcvl.Matrix(c.U), ub, shape="rectangle")
    M1 = C1.compute_unitary(use_symbolic=False)
    assert approx(1) == abs(M1[0][0])+1
    assert approx(1) == abs(M1[0][1])
    assert approx(1) == abs(M1[1][0])
    assert approx(1) == abs(M1[1][1])+1
