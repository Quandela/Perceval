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

from copy import deepcopy, copy
import math

from perceval.utils.logging import get_logger

from .abstract_mitigation import AbstractMitigation
from ..computation import Computation

from perceval.utils import NoiseModel
from perceval.utils.constants import KEY_RESULTS

class DetectorBalancing(AbstractMitigation):
    # Note: we do not know if it behaves correctly with Feef-Forward

    APPLY_MIN_PHOTONS = False
    APPLY_LOGICAL_SELECTION = False

    def __init__(self):
        """
        A mitigation process that adjusts the probabilities of each output state based on the output
        loss and number of photons in each mode.
        """

    def extend_computation(self, computation: Computation, noise: NoiseModel) -> list[Computation]:
        comp = deepcopy(computation)
        comp.command.name = "probs"
        return [comp]

    def _parse_results(self, computation: Computation, results: list[dict], misc: object) -> dict:
        valids = [math.isfinite(v) and v >= 0. and v <= 1. for v in misc.transmitance_ratios_output]
        if not all(valids):
            get_logger().warn("Calibrated detector transmitance ratios invalid values, replaced with 1.0.")
            # raise ValueError("Calibrated detector transmitance ratios invalid values")
        ratios = [ v if valid else 1. for v, valid in zip(misc.transmitance_ratios_output, valids) ]

        if len(ratios) < computation.experiment.m:
            get_logger().warn(
                "Not enough loss ratio for DetectorBalancing: "
                "defaulting missing ones to 1."
            )
            ratios.extend([1.] * (computation.experiment.m - len(ratios)))

        res = copy(results[0])  # We are going to modify this to keep custom fields as much as we can
        #TODO: check what happens if there are less photons than in the input state
        for k in res[KEY_RESULTS].keys():
            res[KEY_RESULTS][k] /= math.prod([ratios[k.photon2mode(i)] for i in range(k.n)])
        res[KEY_RESULTS].normalize()

        return res
