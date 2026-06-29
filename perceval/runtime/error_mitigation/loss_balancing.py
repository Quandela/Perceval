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
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
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

from copy import copy
from math import prod

from perceval.utils import BSDistribution, ConversionHelper, NoiseModel
from perceval.utils.constants import KEY_RESULTS

from ..computation import Computation
from .abstract_mitigation import AbstractMitigation


class LossBalancing(AbstractMitigation):
    """
    Output loss balancing

    Correct biasing in results due to non-uniform loss at the output of the PIC.
    """

    APPLY_MIN_PHOTONS = False
    APPLY_LOGICAL_SELECTION = False

    def extend_computation(
        self,
        computation: Computation,
        noise: NoiseModel
    ) -> list[Computation]:
        return [computation]

    def _parse_results(
        self,
        computation: Computation,
        results: list[dict],
        noise: NoiseModel
    ) -> dict:
        res = copy(results[0])
        ratios = ... # TODO: Get the loss ratios data from QPU

        assert all(ratio > 0 for ratio in ratios), (
            "Bad calibration has led to invalid output transmittances."
        )

        dist = ConversionHelper.convert_to(
            "probs",
            res[KEY_RESULTS],
            **computation.parameters,
        )

        # LossBalancing should be applied before the results are de-compiled
        assert len(ratios) == dist.m, "Loss ratios do not match the distribution lengths."

        balanced = BSDistribution()
        for state, prob in dist.items():
            balanced.add(
                state, prob * prod(
                    (1 / ratios[i]) ** n for i, n in enumerate(state)
                ),
            )

        if len(balanced):
            balanced.normalize()
        res[KEY_RESULTS] = balanced
        return res
