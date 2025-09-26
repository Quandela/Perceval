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

from ._validated_params import AValidatedParam, ValidatedBool, ValidatedFloat
from math import pi


class NoiseModel:
    """
    The NoiseModel class contains all noise parameters which are supported by Perceval. Default value of each
    parameter means "no noise", so a NoiseModel constructed with all default parameters leads to a perfect simulation.

    :param brightness: first lens brightness of a quantum dot
    :param indistinguishability: chance two photons are indistinguishable
    :param g2: g²(0) - second order intensity autocorrelation at zero time delay. This parameter is correlated with how
               often two photons are emitted by the source instead of a single one.
    :param g2_distinguishable: g2-generated photons indistinguishability
    :param transmittance: system-wide transmittance (warning, can interfere with the brightness parameter)
    :param phase_imprecision: maximum precision of the phase shifters (0 means infinite precision)
    :param phase_error: maximum random noise on the phase shifters (in radian)
    """

    # Source parameters
    brightness = ValidatedFloat(0, 1, 1)
    indistinguishability = ValidatedFloat(0, 1, 1)
    g2 = ValidatedFloat(0, 1, 0)
    g2_distinguishable = ValidatedBool(True)

    # System-wide parameter
    transmittance = ValidatedFloat(0, 1, 1)

    # Optical circuit parameter
    phase_imprecision = ValidatedFloat(0, default_value=0)
    phase_error = ValidatedFloat(0, pi, 0)

    def __init__(self,
                 brightness: float = None,
                 indistinguishability: float = None,
                 g2: float = None,
                 g2_distinguishable: bool = None,
                 transmittance: float = None,
                 phase_imprecision: float = None,
                 phase_error: float = None
                 ):
        self.brightness = brightness
        self.indistinguishability = indistinguishability
        self.g2 = g2
        self.g2_distinguishable = g2_distinguishable
        self.transmittance = transmittance
        self.phase_imprecision = phase_imprecision
        self.phase_error = phase_error

    def __deepcopy__(self, memo):
        return NoiseModel(**self.__dict__())

    def __str__(self) -> str:
        return str(self.__dict__())

    def __repr__(self) -> str:
        return str(self.__dict__())

    def __dict__(self) -> dict:
        cls = type(self)
        res = {}
        for attr in dir(self):
            if not attr.startswith("__") and isinstance(cls.__dict__.get(attr), AValidatedParam):
                v = getattr(self, attr)
                if v != cls.__dict__[attr].default_value:
                    res[attr] = v
        return res

    def __eq__(self, other) -> bool:
        cls = type(self)
        for attr in dir(self):
            if not attr.startswith("__") and isinstance(cls.__dict__.get(attr), AValidatedParam):
                if getattr(self, attr) != getattr(other, attr):
                    return False
        return True
