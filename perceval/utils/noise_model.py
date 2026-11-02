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
from ._validated_params import AValidatedParam, ValidatedBool, ValidatedFloat, ValidatedListOrScalar, ValidatedInt
from .constants import TRANSMITTANCE_KEY, G2_KEY, INDISTINGUISHABILITY_KEY
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
        self.brightness: float = brightness
        self.indistinguishability: float = indistinguishability
        self.g2: float = g2
        self.g2_distinguishable: bool = g2_distinguishable
        self.transmittance: float = transmittance
        self.phase_imprecision: float = phase_imprecision
        self.phase_error: float = phase_error

    def __deepcopy__(self, memo):
        return NoiseModel(**self.to_dict())

    def __str__(self) -> str:
        return str(self.to_dict())

    def __repr__(self) -> str:
        return str(self.to_dict())

    def to_dict(self) -> dict:
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

# Note: NoiseModel serialization is in serialization._archive_serialization

class MultiSourceNoiseModel:
    # Do not inherit from NoiseModel so that isinstance(NoiseModel) returns False

    XQ_MAX_MODES = 256  # Maximum number of modes in exqalibur

    # Sources parameters
    brightness = ValidatedListOrScalar(ValidatedFloat(0, 1, 1))
    indistinguishability = ValidatedListOrScalar(ValidatedFloat(0, 1, 1))
    external_indistinguishability = ValidatedListOrScalar(ValidatedFloat(0, 1, 1))
    source_positions = ValidatedListOrScalar(ValidatedInt(0, 125, 0))  # For each mode, which source is connected
    # _g2 = ValidatedListOrScalar(ValidatedFloat(0, 1, 0))

    # System-wide parameter
    # _g2_distinguishable = ValidatedBool(True)  # System wide as the model doesn't allow for separate values
    # _transmittance = ValidatedFloat(0, 1, 1)

    # Optical circuit parameter
    phase_imprecision = ValidatedFloat(0, default_value=0)
    phase_error = ValidatedFloat(0, pi, 0)

    def __init__(self,
                 brightness: list[float] = None,
                 indistinguishability: list[float] = None,
                 external_indistinguishability: list[float] = None,
                 # g2: float = None,
                 source_positions: list[int] = None,
                 # g2_distinguishable: bool = None,
                 # transmittance: float = None,
                 phase_imprecision: float = None,
                 phase_error: float = None
                 ):
        self.brightness: list[float] = brightness
        self.indistinguishability: list[float] = indistinguishability
        self.external_indistinguishability: list[float] = external_indistinguishability
        self.source_positions: list[int] = source_positions if source_positions is not None else [0] * self.XQ_MAX_MODES
        # self.g2 = g2
        # self.g2_distinguishable = g2_distinguishable
        # self.transmittance = transmittance
        self.phase_imprecision: float = phase_imprecision
        self.phase_error: float = phase_error

        # Validation : all values must either be scalars or lists of the same length as the number of sources.
        # Replaces scalars by a repeated list of that length
        if source_positions is not None:
            n_sources = max(self.source_positions) + 1

            if isinstance(self.brightness, (list, tuple)):
                if len(self.brightness) != n_sources:
                    raise ValueError(f"Number of brightness parameters {len(self.brightness)} does not match number of sources {n_sources}")
            else:
                self.brightness = [self.brightness] * n_sources  # Default to same brightness everywhere

            if isinstance(self.indistinguishability, (list, tuple)):
                if len(self.indistinguishability) != n_sources:
                    raise ValueError(f"Number of brightness parameters {len(self.indistinguishability)} does not match number of sources {n_sources}")
            else:
                self.indistinguishability = [self.indistinguishability] * n_sources

            if isinstance(self.external_indistinguishability, (list, tuple)):
                if len(self.external_indistinguishability) != n_sources:
                    raise ValueError(
                        f"Number of brightness parameters {len(self.external_indistinguishability)} does not match number of sources {n_sources}")
            else:
                self.external_indistinguishability = [self.external_indistinguishability] * n_sources

        else:
            n_sources_set = {1}  # 1 for the source positions that default to all 0
            for val in (self.brightness, self.indistinguishability, self.external_indistinguishability):
                if isinstance(val, (list, tuple)):
                    n_sources_set.add(len(val))

            if len(n_sources_set) != 1:
                raise ValueError(f"Received different numbers of sources: {n_sources_set}")
            n_sources = n_sources_set.pop()
            if n_sources != 1:
                raise ValueError(f"The sources positions are not specified - Expected all numbers between 0 and {n_sources - 1}")

            # Here we have 1 source
            self.source_positions = [0] * self.XQ_MAX_MODES

            if not isinstance(self.brightness, (list, tuple)):
                self.brightness = [self.brightness]
            if not isinstance(self.indistinguishability, (list, tuple)):
                self.indistinguishability = [self.indistinguishability]
            if not isinstance(self.external_indistinguishability, (list, tuple)):
                self.external_indistinguishability = [self.external_indistinguishability]

    @property
    def n_sources(self) -> int:
        return len(self.brightness)

    def __deepcopy__(self, memo):
        return MultiSourceNoiseModel(**self.to_dict())

    def __str__(self) -> str:
        return str(self.to_dict())

    def __repr__(self) -> str:
        return str(self.to_dict())

    @property
    def g2(self):
        return 0

    @property
    def transmittance(self):
        return 1

    def to_dict(self) -> dict:
        cls = type(self)
        res = {}
        for attr in dir(self):
            if not attr.startswith("__") and isinstance(cls.__dict__.get(attr), AValidatedParam):
                v = getattr(self, attr)
                if not cls.__dict__[attr].is_default_value(v):
                    res[attr] = v
        return res

    def __eq__(self, other) -> bool:
        cls = type(self)
        for attr in dir(self):
            if not attr.startswith("__") and isinstance(cls.__dict__.get(attr), AValidatedParam):
                if getattr(self, attr) != getattr(other, attr):
                    return False
        return True


def perf_dict_to_noise(perfs: dict[str, float]) -> NoiseModel | MultiSourceNoiseModel:
    # TODO: detect multi-source dict

    nm = NoiseModel()
    if TRANSMITTANCE_KEY in perfs:
        nm.transmittance = perfs[TRANSMITTANCE_KEY] / 100
    if INDISTINGUISHABILITY_KEY in perfs:
        nm.indistinguishability = perfs[INDISTINGUISHABILITY_KEY] / 100
    if G2_KEY in perfs:
        nm.g2 = perfs[G2_KEY] / 100
    return nm


def noise_to_perf_dict(nm: NoiseModel | MultiSourceNoiseModel) -> dict:
    if isinstance(nm, NoiseModel):
        return {
            INDISTINGUISHABILITY_KEY: nm.indistinguishability * 100,
            TRANSMITTANCE_KEY: nm.brightness * nm.transmittance * 100,
            G2_KEY: nm.g2 * 100
        }

    return {
        # TODO
    }
