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

import perceval as pcvl
import perceval.components.unitary_components as comp
from perceval.components import Circuit
from perceval.components.comp_utils import decompose_perms
import perceval.algorithm as algo
import numpy as np
import pytest
import random


def test_permutation_3():
    circuit = comp.PERM([2, 0, 1])
    p = pcvl.Processor("SLOS", circuit)

    ca = algo.Analyzer(p, input_states=[pcvl.BasicState("|1,0,0>")], output_states="*")
    i_001 = ca.col(pcvl.BasicState("|0, 0, 1>"))
    expected_dist = np.zeros(3)
    expected_dist[i_001] = 1
    assert np.allclose(ca.distribution, expected_dist)

    ca = algo.Analyzer(p, input_states=[pcvl.BasicState("|0,1,0>")], output_states="*")
    i_100 = ca.col(pcvl.BasicState("|1, 0, 0>"))
    expected_dist = np.zeros(3)
    expected_dist[i_100] = 1
    assert np.allclose(ca.distribution, expected_dist)

    ca = algo.Analyzer(p, input_states=[pcvl.BasicState("|0,0,1>")], output_states="*")
    i_010 = ca.col(pcvl.BasicState("|0, 1, 0>"))
    expected_dist = np.zeros(3)
    expected_dist[i_010] = 1
    assert np.allclose(ca.distribution, expected_dist)


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


@pytest.mark.parametrize("perm_list", [[2, 0, 1], [2, 3, 1, 0]])
def test_n_mode_permutation_in_2_mode_perms(perm_list):
    n_mode_perm = comp.PERM(perm_list)
    new_circ = n_mode_perm.break_in_2_mode_perms()

    for _, perm in new_circ:
        assert isinstance(perm, comp.PERM)
        assert perm.m == 2

    assert np.all(n_mode_perm.compute_unitary() == new_circ.compute_unitary())


@pytest.mark.parametrize('ith_run', range(10))
def test_random_perm_breakup_run_multiple(ith_run):
    my_perm_list = list(range(15))
    random.shuffle(my_perm_list)
    n_mode_perm = comp.PERM(my_perm_list)

    new_circ = n_mode_perm.break_in_2_mode_perms()
    for _, perm in new_circ:
        assert isinstance(perm, comp.PERM)
        assert perm.m == 2

    assert np.all(n_mode_perm.compute_unitary() == new_circ.compute_unitary())


def test_circuit_decompose_perms():
    # tests if any given circuit with n-mode perm returns a circuit with only 2-mode perms

    c = Circuit(3) // (0, comp.PERM([2, 0, 1])) // (0, comp.BS.H()) // (1, comp.PERM([1, 0]))

    decomp_c = decompose_perms(c)

    assert c.m == decomp_c.m

    for r, c in decomp_c:
        if isinstance(c, comp.PERM):
            assert c.m == 2
