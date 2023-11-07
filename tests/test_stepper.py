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

from perceval.components import BS, PS, GenericInterferometer
from perceval.simulators import Stepper, Simulator
from perceval.utils import BasicState
from perceval.backends import NaiveBackend


def test_stepper_basic_interference():
    c = BS()
    sim = Stepper()
    sim.set_circuit(c)
    res = sim.probs(BasicState([1, 1]))
    assert len(res) == 2
    assert pytest.approx(res[BasicState([2, 0])]) == 0.5
    assert pytest.approx(res[BasicState([0, 2])]) == 0.5


def test_stepper_complex_circuit():
    def _gen_mzi(i: int):
        return BS(BS.r_to_theta(0.42)) // PS(math.pi+i*0.1) // BS(BS.r_to_theta(0.52)) // PS(math.pi/2)

    c = GenericInterferometer(4, _gen_mzi)
    stepper_sim = Stepper()
    stepper_sim.set_circuit(c)
    stepper_res = stepper_sim.probs(BasicState([1, 0, 1, 0]))

    slos_sim = Simulator(NaiveBackend())
    slos_sim.set_circuit(c)
    slos_res = slos_sim.probs(BasicState([1, 0, 1, 0]))

    assert len(stepper_res) == len(slos_res)
    for stepper_bs, stepper_p in stepper_res.items():
        assert slos_res[stepper_bs] == pytest.approx(stepper_p)
