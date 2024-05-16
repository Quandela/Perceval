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
from ..utils.statevector import BSCount
from ._loss_mitigation_utils import _gen_lossy_dists, _get_avg_exp_from_uni_dist



def photon_loss_mitigation(noisy_input: BSCount, ideal_photon_count: int, threshold_stats = False):
    """
    Classical statistical technique to mitigate errors in output svd cause by photon loss

    :param noisy_input:
    :param ideal_photon_count:
    :param threshold_stats: # todo: ask what exactly it would be physically?mathematically for proper naming
    :return
    """
    m = next(iter(noisy_input)).m

    noisy_distributions = _gen_lossy_dists(noisy_input, ideal_photon_count, m, threshold_stats)

    # GET AVERAGE EXPONENT USING AVERAGE DISTANCE FROM UNIFORM PROBABILITY
    z = _get_avg_exp_from_uni_dist(noisy_distributions, m, ideal_photon_count)
    median_of_means = z[0]

    # Generating the mitigated distribution using the decay parameter.
    mitigated_distribution = []
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
            mitigated_distribution.append(c_mn_inv)
        elif noisy_distributions[1][k] < c_mn_inv < noisy_distributions[2][k]:
            mitigated_distribution.append(c_mn_inv)
        else:
            mitigated_distribution.append(func1(0, z[0]))

    mitigated_distribution = [0 if i < 0 else i for i in mitigated_distribution]

    mitigated_distribution = mitigated_distribution / np.sum(mitigated_distribution)

    post_distribution = noisy_distributions[0]
    # todo: ask - use of post-distribution and pattern map
    # todo: confirm -> this is simply a postselected svd from BSSamples?

    # return mitigated_distribution, post_distribution, pattern_map
    return mitigated_distribution
    # this is actually probs values for all states, todo: change to distribution

# todo: discuss use of photon recycled pdf (ref: get_photon_recycling_pdf)
# todo: ask -> metric/measure of efficiency? less lossy?

# an unitary circuit with loss > 0.5. mitigated would be better than post-selected
# mitigating expectation values -> useeful, not here
