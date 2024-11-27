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
from perceval.components import Circuit, BS, PS
from perceval.utils import BasicState, P
from perceval.components.feed_forward_configurator import FFCircuitProvider, FFConfigurator


def test_generic():
    default_circuit = BS()
    m = 3
    offset = 1
    config = FFCircuitProvider(m, offset, default_circuit)

    assert config.default_circuit == default_circuit, "Incorrect default circuit"

    place = (2, 3, 4)

    assert config._max_circuit_size == 2, "Incorrect maximum circuit size"
    assert config.config_modes(place) == (6, 7), "Incorrect place of configured circuit"

    assert config.configure(BasicState(m * [0])) == default_circuit, "Incorrect output circuit"

    config.circuit_offset = -1
    assert config.config_modes(place) == (0, 1), "Incorrect place of configured circuit"


def test_ff_circuit_provider():
    default_circuit = Circuit(2)
    m = 3
    offset = 1

    tested_circuit = BS()

    config = FFCircuitProvider(m, offset, default_circuit)
    config.add_configuration([1, 1, 0], tested_circuit)

    assert config.configure(BasicState(m * [0])) == default_circuit, "Incorrect output default circuit"
    assert config.configure(BasicState([1, 1, 0])) == tested_circuit, "Incorrect output circuit"

    config.reset_map()
    assert config.configure(BasicState([1, 1, 0])) == default_circuit, "Incorrect output circuit"

    config.circuit_map = {BasicState([1, 1, 0]): tested_circuit}
    assert config.configure(BasicState([1, 1, 0])) == tested_circuit, "Incorrect output circuit"


def test_ffconfigurator():
    controlled_circuit = BS.H() // PS(phi=P("phi_a")) // BS.H()
    ffc = FFConfigurator(2, 0, controlled_circuit, default_config={"phi_a": 1})
    assert ffc.circuit_template() == controlled_circuit

    ffc.add_configuration((0, 1), {"phi_a": 0})
    ffc.add_configuration((1, 0), {"phi_a": np.pi})

    c_0_1 = ffc.configure(BasicState([0, 1]))
    assert float(c_0_1._components[1][1].param("phi")) == 0
    assert np.allclose(c_0_1.compute_unitary(), np.eye(2))

    c_1_0 = ffc.configure(BasicState([1, 0]))
    assert float(c_1_0._components[1][1].param("phi")) == np.pi

    c_2_0 = ffc.configure(BasicState([2, 0]))  # Unknown detection triggers default configuration
    assert float(c_2_0._components[1][1].param("phi")) == 1


def test_ffconfigurator_failures():
    controlled_circuit = BS.H(theta=P("theta0")) // PS(phi=P("phi_a")) // BS.H()
    with pytest.raises(NameError):
        FFConfigurator(2, 0, controlled_circuit, default_config={"toto0": 0.5, "phi_c": 1})
    with pytest.raises(NameError):
        FFConfigurator(2, 0, controlled_circuit, default_config={"phi_a": 1, "phi_c": 1})
    with pytest.raises(ValueError):  # Not enough params in config
        FFConfigurator(2, 0, controlled_circuit, default_config={"phi_a": 1})

    ffc = FFConfigurator(2, 0, controlled_circuit, default_config={"theta0": 2, "phi_a": 1})
    with pytest.raises(NameError):
        ffc.add_configuration((0, 1), {"toto0": 0.5, "phi_c": 1})
    with pytest.raises(NameError):
        ffc.add_configuration((0, 1), {"phi_a": 1, "phi_c": 1})
    with pytest.raises(ValueError):  # Not enough params in config
        ffc.add_configuration((0, 1), {"phi_a": 1})
