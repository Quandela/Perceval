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

from perceval import Circuit
from perceval.components import BS
from perceval.utils.statevector import BasicState
from perceval.components.feed_forward_configurator import FFMapper


@pytest.mark.parametrize("configurator_class", [FFMapper])
def test_generic_configurator(configurator_class):
    default_circuit = BS()
    m = 3
    offset = 1
    config = configurator_class(m, offset, default_circuit)

    assert config.default_circuit == default_circuit, "Incorrect default circuit"

    place = (2, 3, 4)

    assert config._max_circuit_size == 2, "Incorrect maximum circuit size"
    assert config.config_modes(place) == (6, 7), "Incorrect place of configured circuit"

    assert config.configure(BasicState(m * [0])) == default_circuit, "Incorrect output circuit"

    config.circuit_offset = -1
    assert config.config_modes(place) == (0, 1), "Incorrect place of configured circuit"


def test_circuit_map_ff_config():
    default_circuit = Circuit(2)
    m = 3
    offset = 1

    tested_circuit = BS()

    config = FFMapper(m, offset, default_circuit)
    config.add_configuration([1, 1, 0], tested_circuit)

    assert config.configure(BasicState(m * [0])) == default_circuit, "Incorrect output default circuit"
    assert config.configure(BasicState([1, 1, 0])) == tested_circuit, "Incorrect output circuit"

    config.reset_map()
    assert config.configure(BasicState([1, 1, 0])) == default_circuit, "Incorrect output circuit"

    config.circuit_map = {BasicState([1, 1, 0]): tested_circuit}
    assert config.configure(BasicState([1, 1, 0])) == tested_circuit, "Incorrect output circuit"
