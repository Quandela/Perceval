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

from exqalibur import BSCount, BSSamples
from .bsdistribution import BSDistribution
from multipledispatch import dispatch


@dispatch(BSDistribution, int)
def apply_min_photons(distribution: BSDistribution, min_photon_filter: int) -> tuple[BSDistribution, float]:
    """
    Applies a filter to remove all states having less than min_photon_filter photons
    :param distribution: The BSDistribution to filter
    :param min_photon_filter: The minimum number of photons needed to filter
    :return: The normalized filtered distribution, and the proportion of accepted states, weighted by their associated probability
    """
    if not min_photon_filter:
        return distribution, 1.

    physical_perf = 1
    new_distribution = BSDistribution()
    for state, prob in distribution.items():
        if state.n >= min_photon_filter:
            new_distribution[state] = prob
        else:
            physical_perf -= prob

    new_distribution.normalize()
    return new_distribution, physical_perf


@dispatch(BSCount, int)
def apply_min_photons(count: BSCount, min_photon_filter: int) -> tuple[BSCount, float]:
    """
    Applies a filter to remove all states having less than min_photon_filter photons
    :param count: The BSCount to filter
    :param min_photon_filter: The minimum number of photons needed to filter
    :return: The filtered BSCount, and the proportion of accepted states, weighted by their associated count
    """
    if not min_photon_filter:
        return count, 1.

    new_count = BSCount()
    accepted = 0
    for state, nb in count.items():
        if state.n >= min_photon_filter:
            new_count[state] = nb
            accepted += nb

    return new_count, accepted / count.total()


@dispatch(BSSamples, int)
def apply_min_photons(samples: BSSamples, min_photon_filter: int) -> tuple[BSSamples, float]:
    """
    Applies a filter to remove all states having less than min_photon_filter photons
    :param samples: The BSSamples to filter
    :param min_photon_filter: The minimum number of photons needed to filter
    :return: The filtered BSSamples, and the proportion of accepted states
    """
    if not min_photon_filter:
        return samples, 1.

    new_samples = BSSamples()
    for sample in samples:
        if sample.n >= min_photon_filter:
            new_samples.append(sample)

    return new_samples, len(new_samples) / len(samples)
