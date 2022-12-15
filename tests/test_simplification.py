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
import numpy as np

import perceval as pcvl
from perceval import Circuit
from perceval.utils.algorithms.simplification import simplify
from perceval.components import unitary_components as comp


def PS_testing(circ, display):
    c2 = simplify(circ, display=display)

    real = []
    for r, c in c2:
        if isinstance(c, comp.PS):
            phi = c.get_variables()["phi"]
            real.append((r[0], phi))

    return real


def test_PS_simp():
    phi = pcvl.P("phi")

    c = (Circuit(3)
         .add(0, comp.PS(np.pi))
         .add(0, comp.PERM([2, 1, 0]))
         .add(0, comp.BS())
         .add(2, comp.PS(phi))
         .add(2, comp.PS(np.pi))
         .add(0, comp.PS(np.pi / 2)))

    expected = [(0, 2 * np.pi), (2, "phi"), (0, np.pi / 2)]
    real = PS_testing(c, True)

    assert real == expected, "PS simplification with display = True not passed"

    expected = [(2, "phi"), (0, np.pi / 2)]
    real = PS_testing(c, False)

    assert real == expected, "PS simplification with display = False not passed"


def PERM_testing(circ, display=False):
    real = []

    c2 = simplify(circ, display=display)

    for r, c in c2:
        if isinstance(c, comp.PERM):
            real.append((r[0], c.perm_vector))
        elif isinstance(c, comp.BS):
            real.append((r[0], c.get_variables()["theta"]))

    return real


def test_perm_simp():
    circ = (Circuit(3)
            .add(0, comp.PERM([0, 2, 1])))

    expected = [(1, [1, 0])]

    real = PERM_testing(circ)

    assert real == expected, "PERM reduction is wrong"

    circ = (Circuit(3)
            .add(0, comp.PERM([0, 2, 1]))
            .add(0, comp.PERM([1, 2, 0])))

    expected = [(0, [1, 0])]
    real = PERM_testing(circ)

    assert real == expected, "PERM reduction is wrong"

    c = (Circuit(3)
         .add(0, comp.PERM([2, 0, 1]))
         .add(0, comp.BS(theta=1))
         .add(0, comp.PERM([1, 2, 0])))

    expected = [(1, 1)]
    real = PERM_testing(c)

    assert real == expected, "PERM simplification moves components wrongly"

    c = (Circuit(4)
         .add(0, comp.PERM([3, 2, 1, 0]))
         .add(0, comp.BS(theta=1))
         .add(2, comp.BS(theta=2))
         .add(0, comp.PERM([3, 2, 1, 0])))

    expected1 = [(0, [1, 0, 3, 2]), (2, 1), (0, 1), (0, [1, 0, 3, 2])]
    expected2 = [(0, [1, 0, 3, 2]), (0, 2), (2, 1), (0, [1, 0, 3, 2])]
    real = PERM_testing(c, True)

    assert real == expected1 or real == expected2, "PERM simplification moves components wrongly"
