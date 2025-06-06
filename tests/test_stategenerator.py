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

import math
import pytest
from unittest.mock import patch

import perceval as pcvl
import networkx as nx

from _test_utils import LogChecker


def test_logical_state_raw():
    sg = pcvl.StateGenerator(pcvl.Encoding.RAW)
    sv = sg.logical_state([0, 1, 1, 0])
    assert str(sv) == "|0,1,1,0>"


def test_logical_state_dual_rail():
    sg = pcvl.StateGenerator(pcvl.Encoding.DUAL_RAIL)
    sv = sg.logical_state([0, 1, 1, 0])
    assert str(sv) == "|1,0,0,1,0,1,1,0>"


def test_logical_state_polarization():
    sg = pcvl.StateGenerator(pcvl.Encoding.POLARIZATION)
    sv = sg.logical_state([0, 1, 1, 0])
    assert str(sv) == "|{P:H},{P:V},{P:V},{P:H}>"


def test_bell_state_raw():
    sg = pcvl.StateGenerator(pcvl.Encoding.RAW)
    sv = sg.bell_state("phi+")
    bsv = pcvl.StateVector(pcvl.BasicState([0, 0])) + pcvl.StateVector(pcvl.BasicState([1, 1]))
    assert sv == bsv
    sv = sg.bell_state("phi-")
    bsv = pcvl.StateVector(pcvl.BasicState([0, 0])) - pcvl.StateVector(pcvl.BasicState([1, 1]))
    assert sv == bsv
    sv = sg.bell_state("psi+")
    bsv = pcvl.StateVector(pcvl.BasicState([0, 1])) + pcvl.StateVector(pcvl.BasicState([1, 0]))
    assert sv == bsv
    sv = sg.bell_state("psi-")
    bsv = pcvl.StateVector(pcvl.BasicState([0, 1])) - pcvl.StateVector(pcvl.BasicState([1, 0]))
    assert sv == bsv


def test_bell_state_dual_rail():
    sg = pcvl.StateGenerator(pcvl.Encoding.DUAL_RAIL)
    sv = sg.bell_state("phi+")
    bsv = pcvl.StateVector(pcvl.BasicState([1, 0, 1, 0])) + pcvl.StateVector(pcvl.BasicState([0, 1, 0, 1]))
    assert sv == bsv
    sv = sg.bell_state("phi-")
    bsv = pcvl.StateVector(pcvl.BasicState([1, 0, 1, 0])) - pcvl.StateVector(pcvl.BasicState([0, 1, 0, 1]))
    assert sv == bsv
    sv = sg.bell_state("psi+")
    bsv = pcvl.StateVector(pcvl.BasicState([1, 0, 0, 1])) + pcvl.StateVector(pcvl.BasicState([0, 1, 1, 0]))
    assert sv == bsv
    sv = sg.bell_state("psi-")
    bsv = pcvl.StateVector(pcvl.BasicState([1, 0, 0, 1])) - pcvl.StateVector(pcvl.BasicState([0, 1, 1, 0]))
    assert sv == bsv


def test_bell_state_polarization():
    sg = pcvl.StateGenerator(pcvl.Encoding.POLARIZATION)
    sv = sg.bell_state("phi+")
    bsv = pcvl.StateVector(pcvl.BasicState("|{P:H},{P:H}>")) + pcvl.StateVector(pcvl.BasicState("|{P:V},{P:V}>"))
    assert sv == bsv
    sv = sg.bell_state("phi-")
    bsv = pcvl.StateVector(pcvl.BasicState("|{P:H},{P:H}>")) - pcvl.StateVector(pcvl.BasicState("|{P:V},{P:V}>"))
    assert sv == bsv
    sv = sg.bell_state("psi+")
    bsv = pcvl.StateVector(pcvl.BasicState("|{P:H},{P:V}>")) + pcvl.StateVector(pcvl.BasicState("|{P:V},{P:H}>"))
    assert sv == bsv
    sv = sg.bell_state("psi-")
    bsv = pcvl.StateVector(pcvl.BasicState("|{P:H},{P:V}>")) - pcvl.StateVector(pcvl.BasicState("|{P:V},{P:H}>"))
    assert sv == bsv


def test_ghz_state_raw():
    sg = pcvl.StateGenerator(pcvl.Encoding.RAW)
    sv = sg.ghz_state(3)
    assert sv == pcvl.StateVector(pcvl.BasicState([0, 0, 0])) + pcvl.StateVector(pcvl.BasicState([1, 1, 1]))
    sv = sg.ghz_state(4)
    assert sv == pcvl.StateVector(pcvl.BasicState([0, 0, 0, 0])) + pcvl.StateVector(pcvl.BasicState([1, 1, 1, 1]))


def test_ghz_state_dual_rail():
    sg = pcvl.StateGenerator(pcvl.Encoding.DUAL_RAIL)
    sv = sg.ghz_state(3)
    assert sv == pcvl.StateVector(pcvl.BasicState([1, 0, 1, 0, 1, 0])) + pcvl.StateVector(
        pcvl.BasicState([0, 1, 0, 1, 0, 1]))
    sv = sg.ghz_state(4)
    assert sv == pcvl.StateVector(pcvl.BasicState([1, 0, 1, 0, 1, 0, 1, 0])) + pcvl.StateVector(
        pcvl.BasicState([0, 1, 0, 1, 0, 1, 0, 1]))


def test_ghz_state_polarization():
    sg = pcvl.StateGenerator(pcvl.Encoding.POLARIZATION)
    sv = sg.ghz_state(3)
    assert sv == pcvl.StateVector(pcvl.BasicState("|{P:H},{P:H},{P:H}>")) + pcvl.StateVector(
        pcvl.BasicState("|{P:V},{P:V},{P:V}>"))
    sv = sg.ghz_state(4)
    assert sv == pcvl.StateVector(pcvl.BasicState("|{P:H},{P:H},{P:H},{P:H}>")) + pcvl.StateVector(
        pcvl.BasicState("|{P:V},{P:V},{P:V},{P:V}>"))


sqrt2_4 = math.sqrt(2) / 4


def test_graph_state_raw():
    sg = pcvl.StateGenerator(pcvl.Encoding.RAW)
    sv = sg.graph_state(nx.path_graph(3))
    assert pytest.approx(sv[pcvl.BasicState('|0,0,0>')]) == sqrt2_4
    assert pytest.approx(sv[pcvl.BasicState('|1,0,0>')]) == sqrt2_4
    assert pytest.approx(sv[pcvl.BasicState('|0,1,0>')]) == sqrt2_4
    assert pytest.approx(sv[pcvl.BasicState('|0,0,1>')]) == sqrt2_4
    assert pytest.approx(sv[pcvl.BasicState('|1,1,0>')]) == -sqrt2_4
    assert pytest.approx(sv[pcvl.BasicState('|1,0,1>')]) == sqrt2_4
    assert pytest.approx(sv[pcvl.BasicState('|0,1,1>')]) == -sqrt2_4
    assert pytest.approx(sv[pcvl.BasicState('|1,1,1>')]) == sqrt2_4


def test_graph_state_dual_rail():
    sg = pcvl.StateGenerator(pcvl.Encoding.DUAL_RAIL)
    sv = sg.graph_state(nx.path_graph(3))
    assert pytest.approx(sv[pcvl.BasicState('|1,0,1,0,1,0>')]) == sqrt2_4
    assert pytest.approx(sv[pcvl.BasicState('|0,1,1,0,1,0>')]) == sqrt2_4
    assert pytest.approx(sv[pcvl.BasicState('|1,0,0,1,1,0>')]) == sqrt2_4
    assert pytest.approx(sv[pcvl.BasicState('|1,0,1,0,0,1>')]) == sqrt2_4
    assert pytest.approx(sv[pcvl.BasicState('|0,1,0,1,1,0>')]) == -sqrt2_4
    assert pytest.approx(sv[pcvl.BasicState('|0,1,1,0,0,1>')]) == sqrt2_4
    assert pytest.approx(sv[pcvl.BasicState('|1,0,0,1,0,1>')]) == -sqrt2_4
    assert pytest.approx(sv[pcvl.BasicState('|0,1,0,1,0,1>')]) == sqrt2_4


def test_graph_state_polarization():
    sg = pcvl.StateGenerator(pcvl.Encoding.POLARIZATION)
    sv = sg.graph_state(nx.path_graph(3))
    assert pytest.approx(sv[pcvl.BasicState('|{P:H},{P:H},{P:H}>')]) == sqrt2_4
    assert pytest.approx(sv[pcvl.BasicState('|{P:V},{P:H},{P:H}>')]) == sqrt2_4
    assert pytest.approx(sv[pcvl.BasicState('|{P:H},{P:V},{P:H}>')]) == sqrt2_4
    assert pytest.approx(sv[pcvl.BasicState('|{P:H},{P:H},{P:V}>')]) == sqrt2_4
    assert pytest.approx(sv[pcvl.BasicState('|{P:V},{P:V},{P:H}>')]) == -sqrt2_4
    assert pytest.approx(sv[pcvl.BasicState('|{P:V},{P:H},{P:V}>')]) == sqrt2_4
    assert pytest.approx(sv[pcvl.BasicState('|{P:H},{P:V},{P:V}>')]) == -sqrt2_4
    assert pytest.approx(sv[pcvl.BasicState('|{P:V},{P:V},{P:V}>')]) == sqrt2_4


def check_dicke_state(state: pcvl.StateVector, n: int, k: int, encoding: pcvl.Encoding):
    state_number = math.comb(k, n)
    assert state_number == len(state)

    photon_number = k
    if encoding == pcvl.Encoding.RAW:
        photon_number = n

    assert all([s_n == photon_number for s_n in state.n])

    if encoding == pcvl.Encoding.DUAL_RAIL:
        assert 2*k == state.m
    else:
        assert state.m == k

    ps = pcvl.PostSelect('&'.join([f"[{i}]<2" for i in range(k)]))
    amp = 1/math.sqrt(state_number)
    for s, a in state:
        assert ps(s)
        assert a == pytest.approx(amp)


@patch.object(pcvl.utils.logging.ExqaliburLogger, "warn")
@pytest.mark.parametrize("encoding", [pcvl.Encoding.RAW, pcvl.Encoding.POLARIZATION, pcvl.Encoding.DUAL_RAIL])
def test_dicke_state(mock_warn, encoding):
    n = 2
    k = 2*n
    ds = pcvl.StateGenerator(encoding).dicke_state(n)
    check_dicke_state(ds, n, k, encoding)
    assert ds == pcvl.StateGenerator(encoding).dicke_state(n, k)
    check_dicke_state(ds, n, k, encoding)

    n = 4
    k = 6
    ds = pcvl.StateGenerator(encoding).dicke_state(n, k)
    check_dicke_state(ds, n, k, encoding)

    with pytest.raises(ValueError):
        pcvl.StateGenerator(encoding).dicke_state(-1)

    with LogChecker(mock_warn):
        assert pcvl.StateGenerator(encoding).dicke_state(4, 2) == pcvl.StateVector()


def test_zero_padded_state():
    n = 2
    m = 5

    state = pcvl.StateGenerator.zero_padded_state(n, m)

    assert state.n == n
    assert state.m == m
    assert state[:n] == pcvl.BasicState(n * [1])

    with pytest.raises(AssertionError):
        pcvl.StateGenerator.zero_padded_state(3, 2)


def test_periodic_state():
    n = 2
    m = 5
    state = pcvl.StateGenerator.periodic_state(n, m)

    assert state == pcvl.BasicState([1, 0, 1, 0, 0])

    with pytest.raises(AssertionError):
        pcvl.StateGenerator.periodic_state(2, 3)


def test_evenly_spaced_state():

    assert pcvl.StateGenerator.evenly_spaced_state(0, 2) == pcvl.BasicState([0, 0])
    assert pcvl.StateGenerator.evenly_spaced_state(1, 5) == pcvl.BasicState([0, 0, 1, 0, 0])
    assert pcvl.StateGenerator.evenly_spaced_state(2, 5) == pcvl.BasicState([1, 0, 0, 0, 1])
    assert pcvl.StateGenerator.evenly_spaced_state(3, 5) == pcvl.BasicState([1, 0, 1, 0, 1])

    # Test with more photons than modes
    assert pcvl.StateGenerator.evenly_spaced_state(8, 5) == pcvl.BasicState([2, 1, 2, 1, 2])
