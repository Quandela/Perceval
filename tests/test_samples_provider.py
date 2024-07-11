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

from perceval.simulators.noisy_sampling_simulator import SamplesProvider
from perceval.backends import Clifford2017Backend
from perceval.components import Circuit, Source
from perceval.utils import BasicState, NoiseModel, BSDistribution


def _svd_to_bsd(svd):
    res = BSDistribution()
    for state, prob in svd.items():
        res.add(state[0], prob)
    return res


def test_samples_provider():
    size = 2
    clifford = Clifford2017Backend()
    clifford.set_circuit(Circuit(size))  # Identity circuit

    ideal_input = BasicState([1]*size)
    noisy_input = Source.from_noise_model(
        NoiseModel(transmittance=0.2, g2=0.05, indistinguishability=0.75)).generate_distribution(ideal_input)
    possible_fock_input = [BasicState([0, 0]), BasicState([1, 0]), BasicState([0, 1]), BasicState([1, 1])]

    provider = SamplesProvider(clifford)
    provider.prepare(_svd_to_bsd(noisy_input), 1000)

    assert provider._pools and provider._weights
    for state in possible_fock_input:
        assert state in provider._pools and state in provider._weights
        res = provider.sample_from(state)
        assert res == state
