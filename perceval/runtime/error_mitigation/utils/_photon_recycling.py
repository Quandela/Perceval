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
from copy import copy
from itertools import combinations, cycle
from scipy.optimize import curve_fit

MAX_LOST_PHOTONS = 2

def check_no_collision(state) -> bool:
    # verifies absence of bunching in a given state
    return all(i <= 1 for i in state)


def _handle_zero_photon_lost_dist(noisy_distributions, pattern_map, noisy_state, count):
    index = pattern_map[tuple(noisy_state)]
    noisy_distributions[0][index] += count


def _handle_one_photon_lost_dist(noisy_distributions, pattern_map, noisy_state, count):
    for t in range(noisy_state.m):  # loop through each bit in string and +1 in each place
        n_ls = list(noisy_state)
        n_ls[t] += 1
        if check_no_collision(n_ls):
            index = pattern_map[tuple(n_ls)]
            noisy_distributions[1][index] += count


def _handle_two_photons_lost_dist(noisy_distributions, pattern_map, noisy_state, count):
    for t in range(noisy_state.m):
        n_ls = list(noisy_state)
        n_ls[t] += 1

        for r in range(t, noisy_state.m):
            n_ls1 = copy(n_ls)
            n_ls1[r] += 1

            if check_no_collision(n_ls1):  # if non-collision is true
                index = pattern_map[tuple(n_ls1)]
                noisy_distributions[2][index] += count


def _generate_one_photon_per_mode_mapping(m, n):
    combos = combinations(range(m), m - n)
    ones_photons = [1] * n
    char_cyc = cycle(ones_photons)
    perms = [tuple(0 if i in combo else next(char_cyc) for i in range(m))
             for combo in combos]
    return {perm: index for perm, index in zip(perms, range(len(perms)))}


def _gen_lossy_dists(noisy_input, ideal_photon_count, pattern_map):
    # Takes non-collision (no bunching) samples as input and
    # outputs approximate distributions for each number of lost photon statistics.
    # MAX_LOST_PHOTONS controls upto how many are considered

    noisy_distributions = [np.zeros(len(pattern_map)) for _ in range(MAX_LOST_PHOTONS + 1)]

    for noisy_state, count in noisy_input.items():  # loop through all the noisy states
        if noisy_state.n < (ideal_photon_count - MAX_LOST_PHOTONS) or not check_no_collision(noisy_state):
            continue
        actual_photon_count = noisy_state.n

        if actual_photon_count == ideal_photon_count:
            _handle_zero_photon_lost_dist(noisy_distributions, pattern_map, noisy_state, count)

        elif actual_photon_count == ideal_photon_count - 1:
            _handle_one_photon_lost_dist(noisy_distributions, pattern_map, noisy_state, count)

        elif actual_photon_count == ideal_photon_count - 2:
            _handle_two_photons_lost_dist(noisy_distributions, pattern_map, noisy_state, count)

    for i in range(MAX_LOST_PHOTONS + 1):
        summed = sum(noisy_distributions[i])
        if summed > 0:
            noisy_distributions[i] = noisy_distributions[i] / summed
    return noisy_distributions


def _get_avg_exp_from_uni_dist(noisy_distributions, m, n):
    # fits a noisy data to provide parameters to generate mitigated distribution
    def func(x, b):
        return uni_value * np.exp(-b * x)

    uniform_prob = 1 / comb(m, n)
    noisy_distributions_from_uni = [np.average(abs(noisy_distribution - uniform_prob))
                                    for noisy_distribution in noisy_distributions]

    uni_value = noisy_distributions_from_uni[0]

    z, _ = curve_fit(f=func,
                     xdata=[0, 1, 2, 50],
                     ydata=[uni_value, noisy_distributions_from_uni[1], noisy_distributions_from_uni[2], 0],
                     bounds=([-5], [5]),
                     maxfev=2000000)

    return z
