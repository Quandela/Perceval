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
from perceval import Processor, Unitary, LC, Matrix, BSDistribution, BasicState, Source
from perceval.algorithm import Sampler
from perceval.simulators.loss_simulator import LossSimulator
from perceval.simulators.simulator import Simulator
from perceval.backends._slos import SLOSBackend


U = Matrix.random_unitary(2)
loss = .3
input_state = BasicState([1, 1])


def test_lc_minimal():
    components = [((0,), LC(loss))]
    expected_svd = BSDistribution()
    expected_svd[BasicState([0])] = loss ** 2
    expected_svd[BasicState([1])] = 2 * loss * (1 - loss)
    expected_svd[BasicState([2])] = (1 - loss) ** 2
    simu = LossSimulator(Simulator(SLOSBackend()))
    simu.set_circuit(components)
    simu.set_min_detected_photon_filter(0)
    res = simu.probs(BasicState([2]))
    assert pytest.approx(res) == expected_svd


def test_lc_commutative():
    # All LC on the input or on the output of the processor yield the same results
    components_1 = [((0, 1), Unitary(U)),
                    ((0,), LC(loss)),
                    ((1,), LC(loss))]

    components_2 = [((0,), LC(loss)),
                    ((1,), LC(loss)),
                    ((0, 1), Unitary(U))]

    simu = LossSimulator(Simulator(SLOSBackend()))
    simu.set_circuit(components_1)
    simu.set_min_detected_photon_filter(0)
    res_1 = simu.probs(input_state)
    simu.set_circuit(components_2)
    res_2 = simu.probs(input_state)
    assert pytest.approx(res_1) == res_2


def test_lc_source_losses_equivalence():
    # When the losses are balanced
    source = Source(losses=loss)
    p = Processor("SLOS", Unitary(U), source)
    p.with_input(input_state)
    p.min_detected_photons_filter(0)

    components_1 = [((0, 1), Unitary(U)),
                    ((0,), LC(loss)),
                    ((1,), LC(loss))]
    simu = LossSimulator(Simulator(SLOSBackend()))
    simu.set_circuit(components_1)
    simu.set_min_detected_photon_filter(0)
    res_1 = simu.probs(input_state)

    sampler = Sampler(p)
    real_out = sampler.probs()["results"]
    assert pytest.approx(real_out) == res_1
