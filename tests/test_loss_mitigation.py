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
import matplotlib.pyplot as plt
import pytest
import numpy as np

from perceval.error_mitigation import photon_loss_mitigation
from perceval.utils import BSCount, BasicState, BSDistribution
from perceval.components import catalog, Source
from perceval.algorithm import Sampler
from perceval import Simulator, SLOSBackend, Processor, Circuit


NOISY_INPUT = BSCount(
        {'|0,0,0,0,0,0,0,0,0>': 10137,
         '|1,0,0,0,0,0,0,0,0>': 1740,
         '|0,1,0,0,0,0,0,0,0>': 900,
         '|0,0,1,0,0,0,0,0,0>': 1717,
        '|0,0,0,1,0,0,0,0,0>': 1537,
         '|0,0,0,0,1,0,0,0,0>': 1470,
         '|0,0,0,0,0,1,0,0,0>': 158,
         '|0,0,0,0,0,0,1,0,0>': 27,
         '|0,0,0,0,0,0,0,1,0>': 221,
         '|0,0,0,0,0,0,0,0,1>': 38,
         '|2,0,0,0,0,0,0,0,0>': 195,
         '|1,1,0,0,0,0,0,0,0>': 52,
         '|1,0,1,0,0,0,0,0,0>': 89,
         '|1,0,0,1,0,0,0,0,0>': 122,
         '|1,0,0,0,1,0,0,0,0>': 146,
         '|1,0,0,0,0,1,0,0,0>': 20,
         '|1,0,0,0,0,0,0,1,0>': 24,
         '|1,0,0,0,0,0,0,0,1>': 5,
         '|0,2,0,0,0,0,0,0,0>': 36,
         '|0,1,1,0,0,0,0,0,0>': 154,
         '|0,1,0,1,0,0,0,0,0>': 78,
         '|0,1,0,0,1,0,0,0,0>': 52,
         '|0,1,0,0,0,1,0,0,0>': 6,
        '|0,1,0,0,0,0,0,1,0>': 14,
        '|0,1,0,0,0,0,0,0,1>': 1,
        '|0,0,2,0,0,0,0,0,0>': 141,
        '|0,0,1,1,0,0,0,0,0>': 70,
        '|0,0,1,0,1,0,0,0,0>': 194,
        '|0,0,1,0,0,1,0,0,0>': 15,
        '|0,0,1,0,0,0,1,0,0>': 2,
        '|0,0,1,0,0,0,0,1,0>': 35,
        '|0,0,1,0,0,0,0,0,1>': 6,
        '|0,0,0,2,0,0,0,0,0>': 124,
        '|0,0,0,1,1,0,0,0,0>': 113,
        '|0,0,0,1,0,1,0,0,0>': 23,
        '|0,0,0,1,0,0,1,0,0>': 5,
        '|0,0,0,1,0,0,0,1,0>': 29,
        '|0,0,0,1,0,0,0,0,1>': 5,
        '|0,0,0,0,2,0,0,0,0>': 104,
        '|0,0,0,0,1,1,0,0,0>': 20,
        '|0,0,0,0,1,0,1,0,0>': 6,
        '|0,0,0,0,1,0,0,1,0>': 6,
        '|0,0,0,0,0,2,0,0,0>': 2,
        '|0,0,0,0,0,1,1,0,0>': 1,
        '|0,0,0,0,0,0,1,1,0>': 2,
        '|3,0,0,0,0,0,0,0,0>': 10,
        '|2,0,1,0,0,0,0,0,0>': 2,
        '|2,0,0,1,0,0,0,0,0>': 6,
        '|2,0,0,0,1,0,0,0,0>': 10,
        '|2,0,0,0,0,0,0,1,0>': 2,
        '|1,2,0,0,0,0,0,0,0>': 1,
        '|1,1,1,0,0,0,0,0,0>': 2,
        '|1,1,0,1,0,0,0,0,0>': 3,
        '|1,1,0,0,1,0,0,0,0>': 5,
        '|1,1,0,0,0,1,0,0,0>': 2,
        '|1,1,0,0,0,0,0,1,0>': 1,
        '|1,0,1,1,0,0,0,0,0>': 2,
        '|1,0,1,0,1,0,0,0,0>': 15,
        '|1,0,0,2,0,0,0,0,0>': 5,
        '|1,0,0,1,1,0,0,0,0>': 7,
        '|1,0,0,1,0,1,0,0,0>': 1,
        '|1,0,0,1,0,0,0,1,0>': 4,
        '|1,0,0,0,2,0,0,0,0>': 2,
        '|1,0,0,0,1,1,0,0,0>': 1,
        '|0,3,0,0,0,0,0,0,0>': 1,
        '|0,2,1,0,0,0,0,0,0>': 8,
        '|0,2,0,0,1,0,0,0,0>': 1,
        '|0,1,2,0,0,0,0,0,0>': 6,
        '|0,1,1,0,1,0,0,0,0>': 2,
        '|0,1,1,0,0,1,0,0,0>': 1,
        '|0,1,1,0,0,0,0,1,0>': 3,
        '|0,1,0,2,0,0,0,0,0>': 5,
        '|0,1,0,0,1,1,0,0,0>': 1,
        '|0,0,3,0,0,0,0,0,0>': 1,
        '|0,0,2,1,0,0,0,0,0>': 5,
        '|0,0,2,0,1,0,0,0,0>': 5,
        '|0,0,2,0,0,0,0,1,0>': 3,
        '|0,0,1,2,0,0,0,0,0>': 1,
        '|0,0,1,0,2,0,0,0,0>': 11,
        '|0,0,1,0,1,1,0,0,0>': 1,
        '|0,0,1,0,1,0,0,1,0>': 3,
        '|0,0,0,3,0,0,0,0,0>': 9,
        '|0,0,0,2,1,0,0,0,0>': 2,
        '|0,0,0,1,2,0,0,0,0>': 3,
        '|0,0,0,1,1,0,0,1,0>': 1,
        '|0,0,0,0,3,0,0,0,0>': 2,
        '|0,0,0,0,2,1,0,0,0>': 1,
        '|0,0,0,0,2,0,1,0,0>': 1})


def tvd(dist_1, dist_2):
    all_keys = set(dist_1.keys()).union(set(dist_1.keys()))
    return sum(abs(dist_1.get(key, 0) - dist_2.get(key, 0)) for key in all_keys)


def test_photon_loss_mitigated():
    # Simply tests a run of recycled mitigation on the NOISY_INPUT BSCount (from Alexia's code)
    mitigated_dist, post_selected_dist = photon_loss_mitigation(NOISY_INPUT, ideal_photon_count=3)

    plt.plot(range(len(mitigated_dist)), mitigated_dist.values(), 'r', label='miti')
    plt.plot(range(len(post_selected_dist)), post_selected_dist.values(), label='ps')
    plt.legend()
    plt.show()

    res = tvd(mitigated_dist, NOISY_INPUT)
    print('tvd miti', res)
    print('tvd ps', tvd(post_selected_dist, NOISY_INPUT))

    assert pytest.approx(sum(mitigated_dist.values()), 1e-5) == 1

def _sampler_setup(output_type: str, lossy: bool):
    pr = Processor("SLOS")
    pr.min_detected_photons_filter(0)
    pr.thresholded_output(True)
    circ = catalog['heralded cnot'].build_circuit()
    pr.set_circuit(circ)

    if lossy:
        src = Source(emission_probability=0.3)
    else:
        src = Source()
    pr.source = src
    input_state = BasicState([0, 1, 0, 1, 1, 1])
    pr.with_input(input_state)

    sampler = Sampler(pr)
    if output_type == 'samples':
        return sampler.sample_count(50000)['results']
    elif output_type == 'probs':
        return sampler.probs()['results']

def _simulator_setup(lossy: bool):

    sim_c = Circuit(m=6)
    circ = catalog['heralded cnot'].build_circuit()
    sim_c.add(0, circ)

    simulator = Simulator(SLOSBackend())
    simulator.set_circuit(sim_c)

    input_state = BasicState([0, 1, 0, 1, 1, 1])

    if lossy:
        input_svd = Source(emission_probability=0.3).generate_distribution(input_state)
    else:
        input_svd = Source().generate_distribution(input_state)

    #
    return simulator.probs_svd(input_svd)['results']


def test_loss_mitigation_on_sampler():
    # Ideal situation
    ideal_dist = _sampler_setup(output_type='probs', lossy=False)

    # Lossy
    lossy_sampling = _sampler_setup(output_type='probs', lossy=True)
    lossy_sampling_2 = BSDistribution()
    for keys, value in lossy_sampling.items():
        if keys.n < 4:
            continue
        else:
            lossy_sampling_2[keys] = value

    lossy_sampling_2.normalize()
    print(lossy_sampling_2)
    # recyling mitigation
    mitigated_dist, post_selected_dist = photon_loss_mitigation(lossy_sampling, ideal_photon_count=4)
    # print(mitigated_dist)
    print(mitigated_dist)

    # print(post_selected_dist)
    # print(tvd(ideal_dist, mitigated_dist))
    # print(tvd(ideal_dist, post_selected_dist))


def test_simple_loss_mitigation():
    # WIP - my idea is to see if I can reproduce with a CNOT
    # create a simple unitary with ideal output dist
    # add loss > 0.5
    # create lossy output dist
    # perform loss mitigation
    # check that it is better than post selected one

    probs = _simulator_setup(lossy=False)
    print('\nideal', len(probs))
    #

    lossy_probs = _simulator_setup(lossy=True)
    print('\nlossy', len(lossy_probs))
    print(lossy_probs)

    mitigated_dist, post_selected_dist = photon_loss_mitigation(lossy_probs, ideal_photon_count=4)
    print('miti',len(mitigated_dist))
    print('ps', len(post_selected_dist))

    #
    plt.plot(range(len(probs)), probs.values(), label='ideal')
    #plt.plot(range(len(lossy_probs)), lossy_probs.values(), label='loss')
    plt.plot(range(len(mitigated_dist)), mitigated_dist.values(), 'r', label='miti')
    plt.plot(range(len(post_selected_dist)), post_selected_dist.values(), label='ps')
    plt.legend()
    plt.show()

    res = tvd(mitigated_dist, probs)
    res_lossy = tvd(lossy_probs, probs)
    print('tvd miti', res)
    print('tvd lossy', res_lossy)
    print('tvd ps', tvd(post_selected_dist, probs))
