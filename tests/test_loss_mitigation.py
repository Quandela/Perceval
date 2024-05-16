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

from perceval.error_mitigation import photon_loss_mitigation
from perceval.utils import BSCount


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

def test_photon_loss_mitigated():
    mitigated_dist = photon_loss_mitigation(NOISY_INPUT, ideal_photon_count=3)
    print('DONE')
    print(mitigated_dist)
    import matplotlib.pyplot as plt
    plt.plot(range(len(mitigated_dist)), mitigated_dist)
    plt.show()
    assert pytest.approx(sum(mitigated_dist), 1e-5) == 1


def test_simple_loss_mitigation():
    # create a simple unitary with ideal output dist
    # add loss > 0.5
    # create lossy output dist
    # perform loss mitigation
    # check that it is better than post selected one
    pass
