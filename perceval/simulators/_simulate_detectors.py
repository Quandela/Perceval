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

from perceval.components.detector import IDetector, DetectionType, get_detection_type
from perceval.utils import BSDistribution, BasicState


def simulate_detectors(dist: BSDistribution, detectors: list[IDetector], min_photons: int = None
                       ) -> tuple[BSDistribution, float]:
    """
    Simulates the effect of imperfect detectors on a theoretical distribution.

    :param dist: A theoretical distribution of detections, as would Photon Number Resolving (PNR) detectors detect.
    :param detectors: A List of detectors
    :param min_photons: Minimum detected photons filter value (when None, does not apply this physical filter)

    :return: A tuple containing the output distribution where detectors were simulated, and a physical performance score
    """
    assert len(detectors) == dist.m, "Mismatch between the number of detectors and the number of modes!"
    detection = get_detection_type(detectors)
    if not dist or detection == DetectionType.PNR:
        return dist, 1

    phys_perf = 1
    result = BSDistribution()
    if detection == DetectionType.Threshold:
        for s, p in dist.items():
            s = s.threshold_detection()
            if min_photons is not None and s.n < min_photons:
                phys_perf -= p
            else:
                result[s] += p
        result.normalize()
        return result, phys_perf

    for s, p in dist.items():
        state_distrib = BSDistribution()
        for photons_in_mode, detector in zip(s, detectors):
            if detector is not None:
                state_distrib *= detector.detect(photons_in_mode)
            else:
                state_distrib *= BasicState([photons_in_mode])
        for s_out, p_out in state_distrib.items():
            if min_photons is not None and s_out.n < min_photons:
                phys_perf -= p * p_out
            else:
                result.add(s_out, p * p_out)
    result.normalize()
    return result, phys_perf


def simulate_detectors_sample(sample: BasicState, detectors: list[IDetector], detection: DetectionType = None
                              ) -> BasicState:
    """
    Simulate detectors effect on one output sample. If multiple possible outcome exist, one is randomly chosen

    :param sample: The sample to simulate detectors on
    :param detectors: A list of detectors (with the same length as the sample)
    :param detection: An optional detection type. Can be recomputed from the detectors list, but it's faster to compute
                      it once and pass it for a list a samples to process
    :return: The output sample where the detector imperfection were applied
    """
    if detection is None:
        detection = get_detection_type(detectors)
    if detection == DetectionType.PNR:
        return sample

    if detection == DetectionType.Threshold:
        return sample.threshold_detection()

    state_distrib = BSDistribution()
    for photons_in_mode, detector in zip(sample, detectors):
        state_distrib *= detector.detect(photons_in_mode)
    out_state = state_distrib.sample(1, non_null=False)[0]
    return out_state
