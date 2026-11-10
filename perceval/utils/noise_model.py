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


class MultiSourceNoiseModel:
    """
    The MultiSourceNoiseModel class is a modification of the NoiseModel class, allowing to simulate several noisy sources.
    Note that this model is tied to specific source places, and cannot be used on Experiments bigger than the given number
    of modes.

    One source can be plugged into several modes, but every mode has exactly one source,
    no matter how many photons are inserted into that mode.

    Example:

    >>> nm = MultiSourceNoiseModel(brightness=[0.05, 0.06],
    >>>                            indistinguishability=[0.9, 0.8],
    >>>                            external_indistinguishability=[1, 0.7],
    >>>                            source_positions=[0, 0, 1, 1])
    >>> # Two sources, one with brightness 0.05 and measured HOM of 0.9, plugged into modes 0 and 1
    >>> # one with brightness 0.06 and measured HOM of 0.8, plugged into modes 2 and 3
    >>> # The measured HOM between photons emitted by each source is sqrt(0.9 * 0.8 * 1 * 0.7)

    :param brightness: first lens brightness of each quantum dot
    :param indistinguishability: chance two photons emitted by the same quantum dot are indistinguishable.
     For each source, the measured HOM would be this number.
    :param external_indistinguishability: chance two photons emitted by two different quantum dots are indistinguishable.
     For each source, the measured HOM would be the square root of the product of all indistinguishability
     and external indistinguishability of the given sources. Note that for two sources, only the product of these matters.
    :param source_positions: For each mode, the index of the source that is plugged
    :param phase_imprecision: maximum precision of the phase shifters (0 means infinite precision)
    :param phase_error: maximum random noise on the phase shifters (in radian)
    """
    # Do not inherit from NoiseModel so that isinstance(NoiseModel) returns False

    XQ_MAX_MODES = 256  # Maximum number of modes in exqalibur

    # Sources parameters
    _brightness = ValidatedListOrScalar(ValidatedFloat(0, 1, 1))
    _indistinguishability = ValidatedListOrScalar(ValidatedFloat(0, 1, 1))
    _external_indistinguishability = ValidatedListOrScalar(ValidatedFloat(0, 1, 1))
    # _g2 = ValidatedListOrScalar(ValidatedFloat(0, 1, 0))

    # For each mode, which source is connected
    source_positions = ValidatedListOrScalar(ValidatedInt(0, 125, 0), accept_scalars=False)

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
        self._brightness = brightness
        self._indistinguishability = indistinguishability
        self._external_indistinguishability = external_indistinguishability
        # self.g2 = g2
        # self.g2_distinguishable = g2_distinguishable
        # self.transmittance = transmittance
        self.phase_imprecision: float = phase_imprecision
        self.phase_error: float = phase_error

        # Validation : all values must either be scalars or lists of the same length as the number of sources.
        # Replaces scalars by a repeated list of that length
        if source_positions is not None:
            self.source_positions: list[int] = source_positions
            self._n_sources = max(self.source_positions) + 1
        else:
            # Here we have 1 source
            self.source_positions = [0] * self.XQ_MAX_MODES
            self._n_sources = 1

        if isinstance(self._brightness, (list, tuple)):
            if len(self._brightness) != self._n_sources:
                raise ValueError(f"Number of brightness parameters {len(self._brightness)} does not match number of sources {self._n_sources}")

        if isinstance(self._indistinguishability, (list, tuple)):
            if len(self._indistinguishability) != self._n_sources:
                raise ValueError(f"Number of brightness parameters {len(self._indistinguishability)} does not match number of sources {self._n_sources}")

        if isinstance(self._external_indistinguishability, (list, tuple)):
            if len(self._external_indistinguishability) != self._n_sources:
                raise ValueError(
                    f"Number of brightness parameters {len(self._external_indistinguishability)} does not match number of sources {self._n_sources}")

    @property
    def brightness(self) -> list[float]:
        if isinstance(self._brightness, (list, tuple)):
            return self._brightness
        return [self._brightness] * self._n_sources

    @brightness.setter
    def brightness(self, value: float | list[float] | None):
        self._brightness = value

    @property
    def indistinguishability(self) -> list[float]:
        if isinstance(self._indistinguishability, (list, tuple)):
            return self._indistinguishability
        return [self._indistinguishability] * self._n_sources

    @indistinguishability.setter
    def indistinguishability(self, value):
        self._indistinguishability = value

    @property
    def external_indistinguishability(self) -> list[float]:
        if isinstance(self._external_indistinguishability, (list, tuple)):
            return self._external_indistinguishability
        return [self._external_indistinguishability] * self._n_sources

    @external_indistinguishability.setter
    def external_indistinguishability(self, value):
        self._external_indistinguishability = value

    @property
    def n_sources(self) -> int:
        return self._n_sources

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
