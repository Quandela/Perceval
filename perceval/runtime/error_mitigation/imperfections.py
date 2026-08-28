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

import dataclasses
from copy import copy

from perceval.utils import NoiseModel
from perceval.utils.constants import KEY_NOISE
from perceval.components import IDetector
from perceval.serialization import Serialization


@dataclasses.dataclass
class Imperfections:
    """
    Dataclass representing every imperfection source that can be mitigated.
    """
    noise: NoiseModel
    detectors: list[IDetector | None]


Serialization.register_class(Imperfections, ["noise", "detectors"], tag="Imperfections")


def update_imperfections_from_results(imperfections: Imperfections, results: dict) -> Imperfections:
    # Tool given to any mitigation if needed.
    # It MUST NOT be used to extend the computation during parsing since we want the original extension
    new_imperfections = copy(imperfections)
    if KEY_NOISE in results:
        new_imperfections.noise = results[KEY_NOISE]
    if "detectors" in results:
        new_imperfections.detectors = results["detectors"]
    return new_imperfections
