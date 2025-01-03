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
from qiskit.quantum_info import Operator
from qiskit import QuantumCircuit

import perceval as pcvl
from perceval.components import catalog, PS, BS, PERM


@pytest.mark.parametrize("gate_name, exptd_phi", [("s", math.pi/2), ("sdag", 3*math.pi/2),
                                       ("t", math.pi/4), ("tdag", 7*math.pi/4),
                                                  ("ph", "phi")])
def test_single_qubit_phase_gates(gate_name, exptd_phi):
    c = catalog[gate_name].build_circuit()

    assert c.m == 2

    for _, comp in c._components:
        assert isinstance(comp, PS)
        assert comp.get_variables()['phi'] == exptd_phi


def test_pauli_y_gate():
    c = catalog["y"].build_circuit()

    assert len(c._components) == 3
    assert isinstance(c._components[0][1], PERM)

    assert isinstance(c._components[1][1], PS)
    assert c._components[1][1].get_variables()['phi'] == math.pi/2

    assert isinstance(c._components[2][1], PS)
    assert c._components[2][1].get_variables()['phi'] == 3*math.pi / 2  # simplification modifies -pi/2 -> 3pi/2


def test_ry_gate():
    c = catalog["ry"].build_circuit(theta=5*math.pi/2)

    assert isinstance(c._components[0][1], BS)
    assert c._components[0][1].get_variables()['theta'] == 5*math.pi/2


def test_rz_gate():
    c = catalog["rz"].build_circuit(theta=math.pi)

    assert isinstance(c._components[0][1], PS)
    assert isinstance(c._components[1][1], PS)

    assert c._components[0][1].get_variables()['phi'] == 3 * math.pi / 2
    assert c._components[1][1].get_variables()['phi'] == math.pi/2


def test_rx_gate():
    # TODO : fix test after confirming BS convention
    c = catalog["rx"].build_circuit(theta=math.pi)
    # pcvl.pdisplay(c)
    print('\n')
    print('same as cqasm')
    print(c.compute_unitary())

    c = BS.Rx(theta=math.pi)
    # pcvl.pdisplay(c)
    print('direct bs of pcvl')
    print(c.compute_unitary())
    import numpy as np
    # m = np.array([[np.c], []])

    qisk_circ = QuantumCircuit(1)
    qisk_circ.rx(math.pi, 0)
    print('qiskit')
    print(Operator(qisk_circ))


@pytest.mark.parametrize("gate_name, lo_comp", [('x', PERM), ('z', PS), ('h', BS)])
def test_single_qubit_gates(gate_name, lo_comp):
    c = catalog[gate_name].build_circuit()
    assert c.m == 2
    for _, comp in c._components:
        assert isinstance(comp, lo_comp)
