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
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import perceval as pcvl
from perceval.components.port import *
from perceval.components.base_components import *
import numpy as np


def act_on_phi(value, obj):
    if value:
        obj.assign({"phi": np.pi/2})
    else:
        obj.assign({"phi": np.pi/4})


def test_digital_converter():
    phi = pcvl.P("phi")
    ps = PS(phi)
    bs = SimpleBS()
    detector = DigitalConverterDetector('I act on phi')
    detector.connect_to(ps, act_on_phi)

    assert detector.is_connected_to(ps)
    assert not detector.is_connected_to(bs)
    assert phi.is_symbolic()

    detector.trigger(True)
    assert not phi.is_symbolic()
    assert float(phi) == np.pi/2

    detector.trigger(False)
    assert not phi.is_symbolic()
    assert float(phi) == np.pi/4
