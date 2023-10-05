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

from perceval.simulators.polarization_simulator import PolarizationSimulator
from perceval.simulators.simulator import Simulator
from perceval.backends import NaiveBackend, BackendFactory
from perceval.components import Circuit, BS, PBS, PERM, PS, PR, HWP
from perceval.utils import BasicState, StateVector
from _test_utils import assert_sv_close
import pytest
import math


def _oracle(mark):
    """Values 0, 1, 2 and 3 for parameter 'mark' respectively mark the elements "00", "01", "10" and "11" of the list."""
    oracle_circuit = Circuit(m=2, name='Oracle')
    # The following dictionnary translates n into the corresponding component settings
    oracle_dict = {0: (1, 0), 1: (0, 1), 2: (1, 1), 3: (0, 0)}
    PC_state, LC_state = oracle_dict[mark]
    # Mode b
    if PC_state == 1:
        oracle_circuit //= HWP(0)
    oracle_circuit.add(0, PR(math.pi/2))
    if LC_state == 1:
        oracle_circuit //= HWP(0)
    # Mode a
    if LC_state == 1:
        oracle_circuit //= (1, HWP(0))
    if PC_state == 1:
        oracle_circuit //= (1, HWP(0))
    return oracle_circuit


def _hwp(xsi):
    hwp = Circuit(m=1)
    hwp.add(0, HWP(xsi)).add(0, PS(-math.pi / 2))
    return hwp


def _grover_circuit(mark):
    init_circuit = (Circuit(m=2, name="Initialization")
                    // _hwp(math.pi / 8)
                    // BS.Ry()
                    // PS(-math.pi))

    inversion_circuit = (Circuit(m=2, name='Inversion')
                         // BS.Ry()
                         // _hwp(math.pi / 4)
                         // BS.Ry())

    detection_circuit = Circuit(m=4, name='Detection')
    detection_circuit.add((0, 1), PBS())
    detection_circuit.add((2, 3), PBS())

    grover_circuit = Circuit(m=4, name='Grover')
    grover_circuit.add(0, init_circuit).add(0, _oracle(mark)).add(0, inversion_circuit)
    grover_circuit.add(1, PERM([1, 0])).add(0, detection_circuit)
    return grover_circuit


def test_grover():
    psimu = PolarizationSimulator(Simulator(NaiveBackend()))
    input_state = BasicState("|{P:H},0,0,0>")

    expected = {
        BasicState([0, 0, 0, 1]): _grover_circuit(0),
        BasicState([0, 0, 1, 0]): _grover_circuit(1),
        BasicState([0, 1, 0, 0]): _grover_circuit(2),
        BasicState([1, 0, 0, 0]): _grover_circuit(3)
    }
    for expected_output, circuit in expected.items():
        psimu.set_circuit(circuit)
        result_dist = psimu.probs(input_state)
        assert len(result_dist) == 1
        assert result_dist[expected_output] == pytest.approx(1)


def test_polarization_evolve():
    psimu = PolarizationSimulator(Simulator(NaiveBackend()))
    input_state = BasicState("|{P:H}>")
    circuit = PR(delta=math.pi/4)
    psimu.set_circuit(circuit)
    sv_out = psimu.evolve(input_state)
    expected = StateVector("|{P:H}>") - StateVector("|{P:V}>")
    assert_sv_close(sv_out, expected)


@pytest.mark.parametrize("backend_name", ["Naive", "SLOS", "MPS"])
def test_polarization_circuit_0(backend_name):
    c = HWP(math.pi/4)
    psimu = PolarizationSimulator(Simulator(BackendFactory.get_backend(backend_name)))
    psimu.set_circuit(c)
    res = psimu.probs(BasicState("|{P:H}>"))
    assert len(res) == 1
    assert res[BasicState("|1>")] == pytest.approx(1)
    assert_sv_close(psimu.evolve(BasicState("|{P:H}>")), complex(0,1)*StateVector('|{P:V}>'))
    assert_sv_close(psimu.evolve(BasicState("|{P:V}>")), complex(0,1)*StateVector('|{P:H}>'))
    assert_sv_close(psimu.evolve(BasicState("|{P:D}>")), complex(0,1)*(StateVector('|{P:H}>') + StateVector('|{P:V}>')))
    # assert str(psimu.evolve(BasicState("|{P:A}>"))) == '|{P:A}>'  # P:A isn't properly dealt with anymore
