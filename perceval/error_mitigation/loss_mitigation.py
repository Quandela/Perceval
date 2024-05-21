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

import numpy as np
from math import comb
from scipy.optimize import curve_fit
from ..utils.statevector import BSCount, BSDistribution, BasicState
from ._loss_mitigation_utils import _gen_lossy_dists, _get_avg_exp_from_uni_dist, _generate_one_photon_per_mode_mapping



def photon_loss_mitigation(noisy_input: BSCount, ideal_photon_count: int, threshold_stats = False) -> tuple:
    """
    Statistical technique to mitigate errors caused by photon loss in the output state distribution

    :param noisy_input: Noisy Basic State Samples
    :param ideal_photon_count: Photon count expected in an ideal loss-less situation
    :param threshold_stats: If True would set a noisy sample with bunching to max 1 photon per state

    :return (loss mitigated distribution, post-selected not mitigated distribution)
    """
    m = next(iter(noisy_input)).m

    pattern_map = _generate_one_photon_per_mode_mapping(m, ideal_photon_count)

    noisy_distributions = _gen_lossy_dists(noisy_input, ideal_photon_count, pattern_map, threshold_stats)

    # GET AVERAGE EXPONENT USING AVERAGE DISTANCE FROM UNIFORM PROBABILITY
    z = _get_avg_exp_from_uni_dist(noisy_distributions, m, ideal_photon_count)
    median_of_means = z[0]

    # Generating the mitigated distribution using the decay parameter.
    mitigated_probs = []
    c_mn_inv = 1 / comb(m, ideal_photon_count)

    def func1(x, a):
        return a * np.exp(-median_of_means * x) + c_mn_inv

    # print("\n Now generate mitigated distributions")

    for k in range(len(noisy_distributions[0])):
        z, _ = curve_fit(func1,
                         [1, 2, 50],
                         [noisy_distributions[1][k], noisy_distributions[2][k], c_mn_inv],
                         bounds=([-5], [5]),
                         maxfev=2000000)
        if noisy_distributions[1][k] > c_mn_inv > noisy_distributions[2][k]:
            mitigated_probs.append(c_mn_inv)
        elif noisy_distributions[1][k] < c_mn_inv < noisy_distributions[2][k]:
            mitigated_probs.append(c_mn_inv)
        else:
            mitigated_probs.append(func1(0, z[0]))

    mitigated_probs = [0 if i < 0 else i for i in mitigated_probs]

    mitigated_probs = mitigated_probs / np.sum(mitigated_probs)

    post_selected_probs = noisy_distributions[0]
    # a post selected case by choosing zero photon loss statistics - not mitigation

    mitigated_distribution = BSDistribution()
    post_selected_distribution = BSDistribution()

    for index, keys in enumerate(pattern_map.keys()):
        state = BasicState(keys)
        mitigated_distribution[state] = mitigated_probs[index]
        post_selected_distribution[state] = post_selected_probs[index]

    return mitigated_distribution, post_selected_distribution
