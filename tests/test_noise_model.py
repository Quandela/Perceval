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

from perceval.utils import NoiseModel
import pytest


BRIGHTNESS_KEY = "brightness"
BRIGHTNESS_DEF = 1
PHASE_IMPRECISION_KEY = "phase_imprecision"
G2_KEY = "g2"
G2_DIST_KEY = "g2_distinguishable"


def test_noise_model_default():
    nm = NoiseModel()
    assert len(nm.__dict__()) == 0
    assert nm.brightness == BRIGHTNESS_DEF
    b = nm[BRIGHTNESS_KEY]
    assert b.is_default  # No value was passed
    brightness_new = 0.1
    b.set(brightness_new)
    assert not b.is_default
    assert nm.brightness == brightness_new

    brightness_new = 0.2
    nm[BRIGHTNESS_KEY].set(brightness_new)
    assert nm.brightness == brightness_new


def test_noise_model_init():
    nm = NoiseModel(phase_imprecision=1e-3, g2=0.05)
    assert PHASE_IMPRECISION_KEY in nm.__dict__()
    assert G2_KEY in nm.__dict__()
    assert BRIGHTNESS_KEY not in nm.__dict__()
    assert not nm[PHASE_IMPRECISION_KEY].is_default
    assert not nm[G2_KEY].is_default
    assert nm[BRIGHTNESS_KEY].is_default


def test_noise_model_errors():
    with pytest.raises(ValueError):
        NoiseModel(brightness=1.1)
    with pytest.raises(ValueError):
        NoiseModel(brightness=-1)
    with pytest.raises(TypeError):
        NoiseModel(brightness="bad type")

    with pytest.raises(ValueError):
        NoiseModel(g2=1.1)
    with pytest.raises(ValueError):
        NoiseModel(g2=-1)
    with pytest.raises(TypeError):
        NoiseModel(g2="bad type")

    with pytest.raises(TypeError):
        NoiseModel(g2_distinguishable=0.7)  # Expects a boolean

    nm = NoiseModel()
    with pytest.raises(ValueError):
        nm.set_value(BRIGHTNESS_KEY, 1.3)

    with pytest.raises(TypeError):
        nm.set_value(G2_DIST_KEY, 1.3)


def test_noise_model_eq():
    nm1 = NoiseModel(brightness=0.15, indistinguishability=0.89, g2=0.01, g2_distinguishable=False)
    nm2 = NoiseModel(brightness=0.15, indistinguishability=0.89, g2=0.01, g2_distinguishable=False)
    assert nm1 == nm2
    nm2.set_value(BRIGHTNESS_KEY, 0.2)
    assert nm1 != nm2
    nm2 = NoiseModel(brightness=0.15, indistinguishability=0.89, g2=0.01)
    assert nm1 != nm2

    nm3 = NoiseModel(brightness=0.01, indistinguishability=0.89, g2=0.15, g2_distinguishable=False)
    assert nm1 != nm3
