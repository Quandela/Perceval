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
import math
from perceval.components import catalog, PS, BS, PERM


@pytest.mark.parametrize("gate_name, exptd_phi", [("s", math.pi / 2), ("sdag", -math.pi / 2),
                                                  ("t", math.pi / 4), ("tdag", -math.pi / 4),
                                                  ("ph", 0.0)])
def test_single_qubit_phase_gates(gate_name, exptd_phi):
    c = catalog[gate_name].build_circuit()

    assert c.m == 2

    for _, comp in c._components:
        assert isinstance(comp, PS)

        assert comp.get_variables()['phi'] == exptd_phi % (2 * math.pi)


def test_pauli_y_gate():
    c = catalog["y"].build_circuit()

    assert len(c._components) == 3
    assert isinstance(c._components[0][1], PERM)

    assert isinstance(c._components[1][1], PS)
    assert c._components[1][1].get_variables()['phi'] == math.pi/2

    assert isinstance(c._components[2][1], PS)
    assert pytest.approx(c._components[2][1].get_variables()['phi']) == - math.pi / 2 % (2 * math.pi)


def test_ry_gate():
    c = catalog["ry"].build_circuit(theta=5*math.pi/2)

    assert isinstance(c._components[0][1], BS)
    assert c._components[0][1].get_variables()['theta'] == 5*math.pi/2


def test_rz_gate():
    c = catalog["rz"].build_circuit(theta=math.pi)

    assert isinstance(c._components[0][1], PS)
    assert isinstance(c._components[1][1], PS)

    assert pytest.approx(c._components[0][1].get_variables()['phi']) == - math.pi / 2 % (2 * math.pi)
    assert c._components[1][1].get_variables()['phi'] == math.pi/2


def test_rx_gate():
    # TODO : PCVL-881
    c = catalog["rx"].build_circuit(theta=-math.pi/4)

    assert isinstance(c._components[0][1], BS)
    assert c._components[0][1].get_variables()['theta'] == math.pi/4


@pytest.mark.parametrize("gate_name, lo_comp", [('x', PERM), ('z', PS), ('h', BS)])
def test_single_qubit_gates(gate_name, lo_comp):
    c = catalog[gate_name].build_circuit()
    assert c.m == 2
    for _, comp in c._components:
        assert isinstance(comp, lo_comp)
