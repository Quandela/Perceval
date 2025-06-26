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
from __future__ import annotations

from typing import Callable

from perceval.runtime import cancel_requested
from perceval.components.detector import IDetector, DetectionType, get_detection_type
from perceval.utils import BSDistribution, BasicState


def heralds_compatible_threshold(s: BasicState, heralds: dict[int, int]):
    for m, v in heralds.items():
        if v and not s[m]:  # Note: this case should not happen if an "at least 1" condition was applied on previous step
            return False
        if v == 0 and s[m]:
            return False
    return True


def compute_distributions(s: BasicState, detectors: list[IDetector], heralds: dict[int, int]) -> list[BSDistribution]:
    distributions = []
    for m, (photons_in_mode, detector) in enumerate(zip(s, detectors)):
        if detector is not None:
            d = detector.detect(photons_in_mode)
            if isinstance(d, BasicState):
                d = BSDistribution(d)

            if m in heralds:
                v = heralds[m]
                state = BasicState([v])
                p = d[state]
                if not p:
                    return []
                d = BSDistribution({state: p})

            distributions.append(d)
        elif m not in heralds or heralds[m] == photons_in_mode:
            distributions.append(BSDistribution(BasicState([photons_in_mode])))

        else:
            return []

    return distributions


def simulate_detectors(dist: BSDistribution, detectors: list[IDetector], min_photons: int = 0,
                       prob_threshold: float = 0, heralds: dict[int, int] = {},
                       progress_callback: Callable = None) -> tuple[BSDistribution, float]:
    """
    Simulates the effect of imperfect detectors on a theoretical distribution.

    :param dist: A theoretical distribution of detections, as would Photon Number Resolving (PNR) detectors detect.
    :param detectors: A List of detectors
    :param min_photons: Minimum detected photons filter value (when None, does not apply this physical filter)
    :param prob_threshold: Filter states that have a probability below this threshold
    :param heralds: A dictionary {mode: expected} that will be used for logical selection.
     Beware the performance can only be considered global (and no longer physical) if not empty
    :param progress_callback: A function with the signature `func(progress: float, message: str)`

    :return: A tuple containing the output distribution where detectors were simulated, and a physical performance score
    """
    assert len(detectors) == dist.m, "Mismatch between the number of detectors and the number of modes!"
    detection = get_detection_type(detectors)
    if not dist or detection == DetectionType.PNR:
        return dist, 1

    phys_perf = 0
    result = BSDistribution()
    lbsd = len(dist)
    if detection == DetectionType.Threshold:
        for idx, (s, p) in enumerate(dist.items()):
            if not heralds_compatible_threshold(s, heralds):
                continue
            s = s.threshold_detection()
            if s.n >= min_photons:
                phys_perf += p
                result[s] += p
            if progress_callback and idx % 250000 == 0:  # Every 250000 states
                progress = (idx + 1) / lbsd
                exec_request = progress_callback(progress, "simulate detectors")
                if cancel_requested(exec_request):
                    raise RuntimeError("Cancel requested")
        if len(result):
            result.normalize()
        return result, phys_perf

    for idx, (s, p) in enumerate(dist.items()):
        if progress_callback and idx % 100000 == 0:  # Every 100000 states
            progress = (idx + 1) / lbsd
            exec_request = progress_callback(progress, "simulate detectors")
            if cancel_requested(exec_request):
                raise RuntimeError("Cancel requested")

        distributions = compute_distributions(s, detectors, heralds)
        if not distributions:
            continue

        state_dist = BSDistribution.list_tensor_product(distributions,
                                prob_threshold=max(prob_threshold, prob_threshold / (10 * p) if p > 0 else prob_threshold))
        # "magic factor" 10 like in the simulator

        for s_out, p_out in state_dist.items():
            if s_out.n >= min_photons:
                prob = p * p_out
                phys_perf += prob
                result.add(s_out, prob)

    if len(result):
        result.normalize()
    return result, phys_perf


def simulate_detectors_sample(sample: BasicState, detectors: list[IDetector], detection: DetectionType = None,
                              heralds: dict[int, int] = {},
                              ) -> BasicState | None:
    """
    Simulate detectors effect on one output sample. If multiple possible outcome exist, one is randomly chosen

    :param sample: The sample to simulate detectors on
    :param detectors: A list of detectors (with the same length as the sample)
    :param detection: An optional detection type. Can be recomputed from the detectors list, but it's faster to compute
                      it once and pass it for a list a samples to process
    :param heralds: A dictionary {mode: expected} that will be used for logical selection.
     If the returned state doesn't fulfil this, this function will early return None.

    :return: The output sample where the detector imperfection were applied, or None if the state is logically rejected for the heralds
    """
    if detection is None:
        detection = get_detection_type(detectors)
    if detection == DetectionType.PNR:
        return sample

    if detection == DetectionType.Threshold:
        if heralds_compatible_threshold(sample, heralds):
            return sample.threshold_detection()
        return None

    out_state = BasicState()
    for m, (photons_in_mode, detector) in enumerate(zip(sample, detectors)):
        state_distrib = detector.detect(photons_in_mode) if detector is not None else BasicState([photons_in_mode])
        if isinstance(state_distrib, BSDistribution):
            state_distrib = state_distrib.sample(1, non_null=False)[0]
        if m in heralds and state_distrib[0] != heralds[m]:
            return None
        out_state *= state_distrib

    return out_state
