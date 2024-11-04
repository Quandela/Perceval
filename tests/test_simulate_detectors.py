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

from perceval.simulators._simulate_detectors import simulate_detectors, simulate_detectors_sample
from perceval.components import Detector, BSLayeredPPNR
from perceval.utils import BSDistribution, BasicState, BSSamples, samples_to_sample_count


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
    assert simulate_detectors(bsd, pnr_detector_list)[0] == bsd

    res, _ = simulate_detectors(bsd, thr_detector_list)
    assert len(res) == 4
    assert res[BasicState([1, 1, 1])] == 0.2
    assert res[BasicState([1, 0, 1])] == 0.4
    assert res[BasicState([0, 0, 1])] == 0.15
    assert res[BasicState([1, 1, 0])] == 0.25

    res, _ = simulate_detectors(bsd, mixed_detector_list)
    assert res[BasicState([1, 1, 1])] == 0.2
    assert res[BasicState([1, 0, 1])] == pytest.approx(0.3 * 0.5 + 0.1)
    assert res[BasicState([2, 0, 1])] == pytest.approx(0.3 * 0.5)
    assert res[BasicState([1, 1, 0])] == pytest.approx(0.25 * 0.5)
    assert res[BasicState([2, 1, 0])] == pytest.approx(0.25 * 0.5)
    assert res[BasicState([0, 0, 1])] == 0.15


def test_simulate_detectors_sample():
    bss_in = BSSamples([BasicState([2, 2, 2]),
                        BasicState([2, 0, 3]),
                        BasicState([1, 1, 1]),
                        BasicState([0, 0, 0])])
    pnr = Detector.pnr()
    thr = Detector.threshold()
    expected = [BasicState([2, 1, 1]),
                BasicState([2, 0, 1]),
                BasicState([1, 1, 1]),
                BasicState([0, 0, 0])]
    for s_in, s_expected in zip(bss_in, expected):
        s_out = simulate_detectors_sample(s_in, [pnr, thr, thr])
        assert s_out == s_expected


def test_simulate_detectors_sample_ppnr():
    # PPNR creates multiple possibilities the detector simulation algo needs to sample from
    bs_in = BasicState([2, 2])
    ppnr_detector = BSLayeredPPNR(1)
    bss_out = BSSamples()
    for i in range(1000):
        bss_out.append(simulate_detectors_sample(bs_in, [ppnr_detector]*2))
    bsc_out = samples_to_sample_count(bss_out)
    assert len(bsc_out) == 4
    assert BasicState([2, 2]) in bsc_out
    assert BasicState([2, 1]) in bsc_out
    assert BasicState([1, 2]) in bsc_out
    assert BasicState([1, 1]) in bsc_out
