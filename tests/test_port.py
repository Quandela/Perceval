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

import pytest
import numpy as np

import perceval as pcvl
from perceval import BasicState
from perceval.components.port import *
from perceval.components.unitary_components import PS, BS
from perceval.utils import Encoding


def act_on_phi(value, obj):
    if value:
        obj.assign({"phi": np.pi / 2})
    else:
        obj.assign({"phi": np.pi / 4})


def test_digital_converter():
    phi = pcvl.P("phi")
    ps = PS(phi)
    ps2 = PS(phi)
    bs = BS()
    detector = DigitalConverterDetector('I act on phi')
    detector.connect_to(ps, act_on_phi)

    assert detector.is_connected_to(ps)
    assert not detector.is_connected_to(bs)
    assert not detector.is_connected_to(ps2)
    assert phi.is_symbolic()

    detector.trigger(True)
    assert not phi.is_symbolic()
    assert float(phi) == np.pi / 2

    detector.trigger(False)
    assert not phi.is_symbolic()
    assert float(phi) == np.pi / 4


def test_basic_state_conversion():
    ports = [Herald(1), Port(Encoding.DUAL_RAIL, "belle"), Port(Encoding.RAW, "bulle"),
             Herald(1), Herald(0), Port(Encoding.TIME, "rebelle"), Herald(1)]

    assert BasicState([0, 1, 0, 1]) == get_basic_state_from_ports(ports, LogicalState([1, 0, 1]))
    with pytest.raises(IndexError):
        get_basic_state_from_ports(ports, LogicalState([1, 0]))
    with pytest.raises(IndexError):
        get_basic_state_from_ports(ports, LogicalState([1, 0, 1, 0]))
    assert BasicState([1, 0, 1, 0, 1, 0, 1, 1]) == get_basic_state_from_ports(
        ports, LogicalState([1, 0, 1]), add_herald_and_ancillary=True)

    assert BasicState([1, 0, 0, 0]) == get_basic_state_from_ports(ports, LogicalState([0, 0, 0]))
    assert BasicState([1, 1, 0, 0, 1, 0, 0, 1]) == get_basic_state_from_ports(
        ports, LogicalState([0, 0, 0]), add_herald_and_ancillary=True)

    assert BasicState([0, 1, 1, 1]) == get_basic_state_from_ports(ports, LogicalState([1, 1, 1]))
    assert BasicState([1, 0, 1, 1, 1, 0, 1, 1]) == get_basic_state_from_ports(
        ports, LogicalState([1, 1, 1]), add_herald_and_ancillary=True)
