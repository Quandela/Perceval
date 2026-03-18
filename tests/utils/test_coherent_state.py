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

from perceval import BS, Processor, SLOSBackend, Simulator
from perceval.utils.coherent_state import CoherentState


def test_empty_state():

    empty = CoherentState()

    assert empty.m == 0
    assert empty.n == 0
    assert str(empty) == "|>"

    assert empty == CoherentState()

    assert not empty.has_annotations
    assert not empty.has_polarization

def test_coherent_state():

    state = CoherentState([1, 2+3.4j])

    assert str(state) == "|1+0j, 2+3.4j>"
    assert state.m == 2
    assert state.n == 1

    other = CoherentState([3.1, 0.2])

    assert state.merge(other) == CoherentState([4.1, 2.2+3.4j])

    assert state * other == CoherentState([1, 2+3.4j, 3.1, 0.2])

    small_state = CoherentState([1.1j])

    assert state.set_slice(small_state, 0, 1) == CoherentState([1.1j, 2+3.4j])
    assert state.set_slice(small_state, 1, 2) == CoherentState([1, 1.1j])

    assert len(state) == state.m

    assert state ** 2 == CoherentState([1, 2+3.4j, 1, 2+3.4j])

    assert state.remove_modes([1]) == CoherentState([1])

    assert state.to_power() == [1, 4 + 3.4 ** 2]

def test_simulation():
    state = CoherentState([1-0.4j, 2+3.4j])

    tot_power = sum(state.to_power())

    p = Processor("SLOS", BS())
    p.with_input(state)

    res_coherent = p.probs()["results"]

    assert pytest.approx(sum(res_coherent.to_power())) == tot_power

    # Now compare with a StateVector simulation
    sv = state.to_state_vector()

    backend = SLOSBackend()
    simulator = Simulator(backend)
    simulator.set_circuit(BS())

    res_sv = simulator.evolve(sv)

    for fs, ampli in res_sv:
        m = fs.photon2mode(0)
        assert pytest.approx(ampli * tot_power ** .5) == res_coherent[m]
