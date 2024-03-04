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

from ._validated_params import AValidatedParam, ValidatedBool, ValidatedFloat
from typing import Dict


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
    """

    def __init__(self,
                 brightness: float = None,
                 indistinguishability: float = None,
                 g2: float = None,
                 g2_distinguishable: bool = None,
                 transmittance: float = None,
                 phase_imprecision: float = None
                 ):
        # Source parameters
        self._params: Dict[str, AValidatedParam] = {}
        self._add_param(ValidatedFloat("brightness", brightness, 0, 1, 1))
        self._add_param(ValidatedFloat("indistinguishability", indistinguishability, 0, 1, 1))
        self._add_param(ValidatedFloat("g2", g2, 0, 1, 0))
        self._add_param(ValidatedBool("g2_distinguishable", g2_distinguishable, True))

        # System-wide parameter
        self._add_param(ValidatedFloat("transmittance", transmittance, 0, 1, 1))

        # Optical circuit parameter
        self._add_param(ValidatedFloat("phase_imprecision", phase_imprecision, 0, default_value=0))

    def _add_param(self, param: AValidatedParam):
        self._params[param.name] = param
        cls = type(self)
        if not hasattr(cls, param.name):
            # Create a property named after the param name
            setattr(cls, param.name, property(lambda slf: slf._params[param.name].get()))

    def __getitem__(self, param_name: str) -> AValidatedParam:
        if param_name in self._params:
            return self._params[param_name]
        raise KeyError(f"No parameter named '{param_name}'")

    def set_value(self, param_name: str, value):
        self[param_name].set(value)

    def __str__(self) -> str:
        return str(self.__dict__())

    def __dict__(self) -> dict:
        return {k: v.get() for k, v in self._params.items() if not v.is_default}

    def __eq__(self, other) -> bool:
        if len(self._params) != len(other._params):
            return False
        for field in self._params:
            if self._params[field] != other._params[field]:
                return False
        return True
