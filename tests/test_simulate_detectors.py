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

from perceval.simulators._simulate_detectors import simulate_detectors
from perceval.components import Detector, BSLayeredPPNR
from perceval.utils import BSDistribution, BasicState


def test_simulate_detectors():
    pnr_detector_list = [Detector.pnr()] * 3  # Only PNR detectors
    thr_detector_list = [Detector.threshold()] * 3  # Only threshold detectors
    mixed_detector_list = [BSLayeredPPNR(1), Detector.pnr(), Detector.threshold()]

    bsd = BSDistribution({
        BasicState([1, 1, 1]): 0.2,
        BasicState([2, 0, 1]): 0.3,
        BasicState([2, 1, 0]): 0.25,
        BasicState([0, 0, 3]): 0.15,
        BasicState([1, 0, 2]): 0.1
    })
    assert simulate_detectors(bsd, pnr_detector_list) == bsd

    res = simulate_detectors(bsd, thr_detector_list)
    assert len(res) == 4
    assert res[BasicState([1, 1, 1])] == 0.2
    assert res[BasicState([1, 0, 1])] == 0.4
    assert res[BasicState([0, 0, 1])] == 0.15
    assert res[BasicState([1, 1, 0])] == 0.25

    res = simulate_detectors(bsd, mixed_detector_list)
    assert res[BasicState([1, 1, 1])] == 0.2
    assert res[BasicState([1, 0, 1])] == pytest.approx(0.3 * 0.5 + 0.1)
    assert res[BasicState([2, 0, 1])] == pytest.approx(0.3 * 0.5)
    assert res[BasicState([1, 1, 0])] == pytest.approx(0.25 * 0.5)
    assert res[BasicState([2, 1, 0])] == pytest.approx(0.25 * 0.5)
    assert res[BasicState([0, 0, 1])] == 0.15
