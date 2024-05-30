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

from perceval.simulators import NoisySamplingSimulator
from perceval.backends import Clifford2017Backend
from perceval.components import Unitary, Source
from perceval.utils import Matrix, BasicState, SVDistribution

import pytest


@pytest.mark.parametrize("max_samples, max_shots", [(100, None), (100, 10), (10, 100)])
def test_perfect_sampling(max_samples, max_shots):
    size = 8
    c = Unitary(Matrix.random_unitary(size))
    sim = NoisySamplingSimulator(Clifford2017Backend())
    sim.set_circuit(c)
    input_state = SVDistribution(BasicState([1, 0]*4))
    sampling = sim.samples(input_state, max_samples, max_shots)
    assert sampling['physical_perf'] == 1
    assert sampling['logical_perf'] == 1
    assert len(sampling['results']) == max_samples if max_shots is None else min(max_samples, max_shots)
    for output_state in sampling['results']:
        assert output_state.n == 4


def _build_noisy_simulator(size: int):
    c = Unitary(Matrix.random_unitary(size))
    sim = NoisySamplingSimulator(Clifford2017Backend())
    sim.set_circuit(c)
    return sim


def test_sample_0_samples():
    sim = _build_noisy_simulator(6)
    source = Source(losses=0.8, indistinguishability=0.75, multiphoton_component=0.05)
    input_state = source.generate_distribution(BasicState([1, 0] * 3))
    sampling = sim.samples(input_state, 0)
    assert len(sampling['results']) == 0


def test_noisy_sampling():
    sim = _build_noisy_simulator(6)
    source = Source(losses=0.8, indistinguishability=0.75, multiphoton_component=0.05)
    input_state = source.generate_distribution(BasicState([1, 0] * 3))
    sampling = sim.samples(input_state, 100)
    assert sampling['physical_perf'] == 1
    assert sampling['logical_perf'] == 1
    assert len(sampling['results']) == 100

    sim.set_min_detected_photon_filter(2)
    sampling = sim.samples(input_state, 100)
    assert sampling['physical_perf'] < 1
    assert sampling['logical_perf'] == 1
    assert len(sampling['results']) == 100

    # test sample_count too
    sampling = sim.sample_count(input_state, 100)
    assert sampling['physical_perf'] < 1
    assert sampling['logical_perf'] == 1
    assert sampling['results'].total() == 100


def test_noisy_sampling_with_heralds():
    sim = _build_noisy_simulator(6)
    source = Source(losses=0.8, indistinguishability=0.75, multiphoton_component=0.05)
    input_state = source.generate_distribution(BasicState([1, 0] * 3))

    sim.set_min_detected_photon_filter(2)
    sim.set_selection(heralds={0: 0})
    sampling = sim.samples(input_state, 100)
    assert sampling['physical_perf'] < 1
    assert sampling['logical_perf'] < 1
    assert len(sampling['results']) == 100
    for output_state in sampling['results']:
        assert len(output_state) == 6
        assert output_state[0] == 0  # Fixed by the heralding

    sim.keep_heralds(False)
    sampling = sim.sample_count(input_state, 100)
    assert sampling['logical_perf'] < 1
    for output_state in sampling['results']:
        assert len(output_state) == 5  # The ancillary mode was removed from all output states
