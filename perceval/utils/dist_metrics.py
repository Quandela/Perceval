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

from perceval.utils import BSDistribution
from perceval.utils.logging import get_logger, channel


def tvd_dist(dist1: BSDistribution, dist2: BSDistribution) -> float:
    """
    Computes the Total Variation Distance (TVD) between two input BSDistributions.

    :param dist1: First BSDistribution
    :param dist2: Second BSDistribution
    :return : total variation distance between the two BSDistributions (value between 0 and 1)
    """
    common_bs = set(dist1.keys()).intersection(dist2.keys())

    if not common_bs:
        raise ValueError('There are no common BasicStates between the two input distributions. '
                         'Cannot compute TVD')

    if common_bs != set(dist1.keys()) or common_bs != set(dist2.keys()):
        get_logger().warn(f"Distributions have mismatched number of states. {len(common_bs)} common states found "
                          f"and used to compute TVD", channel.user)

    tvd = 0.5 * sum(abs(dist1[basicstate]-dist2[basicstate]) for basicstate in common_bs)

    return tvd


def chi2_distance_dist(dist1: BSDistribution, dist2: BSDistribution) -> float:
    """
    Computes the Chi Squared Distance (TVD) between two input BSDistributions.

    :param dist1: First BSDistribution
    :param dist2: Second BSDistribution
    :return : Chi squared distance between the two BSDistributions
    """
    common_bs = set(dist1.keys()).intersection(dist2.keys())

    if not common_bs:
        raise ValueError('There are no common BasicStates between the two input distributions. '
                         'Cannot compute Chi2 distance')

    if common_bs != set(dist1.keys()) or common_bs != set(dist2.keys()):
        get_logger().warn(f"Distributions have mismatched number of states. {len(common_bs)} common states found "
                          f"and used to compute Chi2 distance", channel.user)

    chi2_dist = 0.5 * sum([(dist1[basicstate] - dist2[basicstate])**2 / (dist1[basicstate] + dist2[basicstate])
                    for basicstate in common_bs])

    return chi2_dist
