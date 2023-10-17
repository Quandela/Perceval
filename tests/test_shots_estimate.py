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

from perceval.runtime import RemoteProcessor
from perceval.components.abstract_processor import AProcessor
from perceval.components import Unitary, BS, PS
from perceval.utils import Matrix, BasicState, P
import random


class _MockRemoteProcessor(RemoteProcessor):
    def __init__(self):
        AProcessor.__init__(self)  # Avoid RemoteProcessor __init__ to prevent the https request from firing
        self._specs = {}


def test_shots_estimate_trivial_filter_values():
    rp = _MockRemoteProcessor()
    m = Matrix.random_unitary(10)
    rp.set_circuit(Unitary(m))
    rp.with_input(BasicState([1]*5 + [0]*5))
    rp.min_detected_photons_filter(1)

    ANY_VALUE = random.randint(1000, 9999999999)

    # with min_detected_photons_filter set to 1, shots and samples are the same
    assert rp.estimate_expected_samples(ANY_VALUE) == ANY_VALUE
    assert rp.estimate_required_shots(ANY_VALUE) == ANY_VALUE

    rp.min_detected_photons_filter(0)
    # same with 0
    assert rp.estimate_expected_samples(ANY_VALUE) == ANY_VALUE
    assert rp.estimate_required_shots(ANY_VALUE) == ANY_VALUE

    # with a filter too high, there's no estimate
    rp.min_detected_photons_filter(6)
    assert rp.estimate_expected_samples(ANY_VALUE) == 0
    assert rp.estimate_required_shots(ANY_VALUE) is None


def test_shots_estimate_regular_use_case():
    rp = _MockRemoteProcessor()
    c = BS() // PS(phi=0.2) // BS()
    rp.set_circuit(c)
    rp.with_input(BasicState([1, 1]))
    assert 28 < rp.estimate_expected_samples(1000) < 32
    assert 32000 < rp.estimate_required_shots(1000) < 33000


def test_shots_estimate_circuit_with_variables():
    rp = _MockRemoteProcessor()
    c = BS() // PS(phi=P("my_phase")) // BS()
    rp.set_circuit(c)
    rp.with_input(BasicState([1, 1]))
    assert 28 < rp.estimate_expected_samples(1000, {"my_phase": 0.2}) < 32
    assert 32000 < rp.estimate_required_shots(1000, {"my_phase": 0.2}) < 33000
