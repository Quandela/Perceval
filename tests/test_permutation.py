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
import perceval.components.base_components as comp
import perceval.algorithm as algo


def test_permutation_3():
    circuit = comp.PERM([2, 0, 1])
    pf = pcvl.get_platform("SLOS")
    ca = algo.Analyzer(pf, circuit, input_states=[pcvl.BasicState("|1,0,0>")], output_states="*")
    assert ca.output_states_list[2] == pcvl.BasicState("|0, 0, 1>")
    assert not((ca.distribution[0]-[0, 0, 1]).any())
    ca = algo.Analyzer(pf, circuit, input_states=[pcvl.BasicState("|0,1,0>")], output_states="*")
    assert ca.output_states_list[0] == pcvl.BasicState("|1, 0, 0>")
    assert not((ca.distribution[0]-[1, 0, 0]).any())
    ca = algo.Analyzer(pf, circuit, input_states=[pcvl.BasicState("|0,0,1>")], output_states="*")
    assert ca.output_states_list[1] == pcvl.BasicState("|0, 1, 0>")
    assert not((ca.distribution[0]-[0, 1, 0]).any())


def test_permutation_inverse():
    perm_vector = [4, 1, 3, 5, 2, 0]
    perm = comp.PERM(perm_vector)
    assert perm.perm_vector == perm_vector
    perm.inverse(h=True)
    assert perm.perm_vector == [perm_vector.index(i) for i in range(len(perm_vector))]
    perm.inverse(h=True)  # Get back to the initial state
    assert perm.perm_vector == perm_vector
    perm.inverse(v=True)
    # Vertical inversion acts as if mode indexes were reversed:
    expected = [max(perm_vector)-i for i in perm_vector]
    expected.reverse()
    assert perm.perm_vector == expected
