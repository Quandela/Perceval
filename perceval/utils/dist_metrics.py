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


def tvd_dist(dist_lh: BSDistribution, dist_rh: BSDistribution) -> float:
    """
    Computes the Total Variation Distance (TVD) between two input BSDistributions.

    :param dist_lh: First BSDistribution
    :param dist_rh: Second BSDistribution
    :return : total variation distance between the two BSDistributions (value between 0 and 1)
    """
    only_dist_lh_states = set(dist_lh.keys()) - set(dist_rh.keys())
    only_dist_rh_states = set(dist_rh.keys()) - set(dist_lh.keys())

    if only_dist_rh_states or only_dist_lh_states:
        get_logger().warn("Some Basic states are missing in one or both of the two input distributions. "
                          "Their values will be set to 0 before computing TVD.", channel.user)

    all_states = set(dist_lh.keys()).union(dist_rh.keys())
    tvd = 0.5 * sum(abs(dist_lh.get(basic_state, 0) - dist_rh.get(basic_state, 0)) for basic_state in all_states)

    return tvd
