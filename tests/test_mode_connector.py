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

import pytest

from perceval.components._mode_connector import ModeConnector, UnavailableModeException, \
    InvalidMappingException
from perceval.components import Processor, Circuit, Port, Encoding, PortLocation


def test_connection_resolver_init():
    in_modes = 6
    p1 = Processor(8)
    p2 = Processor(in_modes)
    connector = ModeConnector(p1, p2, {})
    assert not connector._r_is_component
    assert connector._n_modes_to_connect == in_modes

    circuit = Circuit(in_modes)
    connector = ModeConnector(p1, circuit, {})
    assert connector._r_is_component
    assert connector._n_modes_to_connect == in_modes


def test_connection_int():
    p1 = Processor(6)
    p2 = Processor(4)
    connector = ModeConnector(p1, p2, 0)
    assert connector.resolve() == {0: 0, 1: 1, 2: 2, 3: 3}
    connector = ModeConnector(p1, p2, 1)
    assert connector.resolve() == {1: 0, 2: 1, 3: 2, 4: 3}
    connector = ModeConnector(p1, p2, 2)
    assert connector.resolve() == {2: 0, 3: 1, 4: 2, 5: 3}

    with pytest.raises(UnavailableModeException):
        connector = ModeConnector(p1, p2, -1)
        connector.resolve()


def test_connection_dict_int():
    p1 = Processor(8)
    p2 = Processor(6)
    valid_mapping = {0: 2, 1: 4, 2: 5, 3: 0, 4: 1, 5: 3}
    connector = ModeConnector(p1, p2, valid_mapping)
    assert connector.resolve() == valid_mapping

    invalid_mapping = {-1: 2, 1: 4, 2: 5, 3: 0, 4: 1, 5: 3}  # -1 is invalid
    connector = ModeConnector(p1, p2, invalid_mapping)
    with pytest.raises(UnavailableModeException):
        connector.resolve()

    invalid_mapping = {0: 2, 1: 4, 2: 5, 3: 0, 4: 1, 5: 3, 6: 6}  # mapping too big
    connector = ModeConnector(p1, p2, invalid_mapping)
    with pytest.raises(InvalidMappingException):
        connector.resolve()


def test_connection_dict_str():
    """Test with port names"""
    p1 = Processor(4)
    p1.add_port(0, Port(Encoding.dual_rail, "q0"), PortLocation.output)
    p1.add_port(2, Port(Encoding.dual_rail, "q1"), PortLocation.output)
    p2 = Processor(4)
    p2.add_port(0, Port(Encoding.dual_rail, "in_A"), PortLocation.input)
    p2.add_port(2, Port(Encoding.dual_rail, "in_B"), PortLocation.input)

    connector = ModeConnector(p1, p2, {'q0': 'in_A', 'q1': 'in_B'})
    assert connector.resolve() == {0: 0, 1: 1, 2: 2, 3: 3}

    connector = ModeConnector(p1, p2, {'q0': 'in_B', 'q1': 'in_A'})
    assert connector.resolve() == {0: 2, 1: 3, 2: 0, 3: 1}

    connector = ModeConnector(p1, p2, {'q0': [2, 3], 'q1': [0, 1]})
    assert connector.resolve() == {0: 2, 1: 3, 2: 0, 3: 1}

    connector = ModeConnector(p1, p2, {'q0': 1, 'q1': 'in_B'})
    with pytest.raises(InvalidMappingException):
        connector.resolve()  # imbalanced size (size of q0 is 2)

    connector = ModeConnector(p1, p2, {'q0': 'bad name', 'q1': 'in_B'})
    with pytest.raises(InvalidMappingException):
        connector.resolve()  # unknown port 'bad name'

    connector = ModeConnector(p1, p2, {'q0': 'in_A', 'bad name': 'in_B'})
    with pytest.raises(InvalidMappingException):
        connector.resolve()  # unknown port 'bad name'

    connector = ModeConnector(p1, p2, {'q0': 'in_A', 'q1': 'in_A'})
    with pytest.raises(InvalidMappingException):
        connector.resolve()  # duplicates

    connector = ModeConnector(p1, p2, {'q0': 'in_A'})
    with pytest.raises(InvalidMappingException):
        connector.resolve()  # mapping too small
