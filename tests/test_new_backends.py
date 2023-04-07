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

from perceval.backends._clifford2017 import Clifford2017Backend
from perceval.backends._naive import NaiveBackend, AProbAmpliBackend
from perceval.components import BS, PS, Circuit
from perceval.utils import BSCount, BasicState
import pytest
import numpy as np


def test_clifford_bs():
    cliff_bs = Clifford2017Backend()
    cliff_bs.set_circuit(BS.H())
    cliff_bs.set_input_state(BasicState([0, 1]))
    counts = BSCount()
    for _ in range(10000):
        counts[cliff_bs.sample()] += 1
    assert 4750 < counts[BasicState("|0,1>")] < 5250
    assert 4750 < counts[BasicState("|1,0>")] < 5250


def check_output_distribution(backend: AProbAmpliBackend, input_state: BasicState, expected: dict):
    backend.set_input_state(input_state)
    prob_list = []
    for (output_state, prob) in backend.prob_distribution().items():
        prob_expected = expected.get(output_state)
        if prob_expected is None:
            assert pytest.approx(0) == prob, "cannot find: %s (prob=%f)" % (str(output_state), prob)
        else:
            assert pytest.approx(prob_expected) == prob, "incorrect value for %s: %f/%f" % (str(output_state),
                                                                                            prob,
                                                                                            prob_expected)
        prob_list.append(prob)
    assert pytest.approx(sum(prob_list)) == 1


def test_probampli_backends():
    for backend_type in [NaiveBackend]:
        backend = backend_type()
        circuit = Circuit(3) // BS.H() // (1, PS(np.pi/4)) // (1, BS.H())
        backend.set_circuit(circuit)
        check_output_distribution(
            backend,
            BasicState("|0,1,1>"),
            {
                BasicState("|0,1,1>"): 0,
                BasicState("|1,1,0>"): 0.25,
                BasicState("|1,0,1>"): 0.25,
                BasicState("|2,0,0>"): 0,
                BasicState("|0,2,0>"): 0.25,
                BasicState("|0,0,2>"): 0.25,
            })
