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

import math
import pytest
from perceval.components import (catalog, Circuit, BS, PS, PERM, Processor, Detector, UnavailableModeException,
                                 FFConfigurator, FFCircuitProvider, Unitary, Barrier)
from perceval.utils import Matrix, P, LogicalState
from perceval.runtime import RemoteProcessor
from _mock_rpc_handler import get_rpc_handler


def test_processor_composition():
    p = catalog['postprocessed cnot'].build_processor()  # Circuit with [0,1] and [2,3] post-selection conditions
    p.add((0, 1), BS())  # Composing with a component on modes [0,1] should work
    with pytest.raises(AssertionError):
        p.add((1, 2), BS())  # Composing with a component on modes [1,2] should fail
    p_bs = Processor("SLOS", BS())
    p.add((0, 1), p_bs)  # Composing with a processor on modes [0,1] should work
    with pytest.raises(AssertionError):
        p.add((1, 2), p_bs)  # Composing with a processor on modes [1,2] should fail


def test_composition_error_post_selection():
    processor = catalog['postprocessed cnot'].build_processor()
    # Composing 2 CNOTs on the exact same modes should work in theory, but not in the current implementation,
    # it's still possible to apply a PostSelect manually to the resulting Processor.
    with pytest.raises(AssertionError):
        processor.add(0, processor)

    processor2 = Processor("SLOS", 5)
    pp_cnot = catalog['postprocessed cnot'].build_processor()
    processor2.add(0, pp_cnot)
    # It's 100% valid that this 2nd case is blocked
    with pytest.raises(AssertionError):
        processor2.add(1, pp_cnot)


def test_processor_composition_mismatch_modes():
    # tests composing a smaller processor into larger one works
    # without breaking simplification (verifies it works with gates based circuits too)
    def sub_size_processor():
        h_cnot = catalog['heralded cnot'].build_processor()
        p = Processor('SLOS', m_circuit=4, name='my_example')
        p.add(0, BS.H())
        p.add(0, h_cnot)
        p.add(1, PS(math.pi / 4))
        p.add(0, h_cnot)
        return p

    smaller_processor = sub_size_processor()
    p = Processor('SLOS', m_circuit=5, name='to_Which_i_add')
    p.add(0, smaller_processor)

    assert len(p.components) == 7  # 3 PERMs get added because heralds need to move

    r_list = []
    comp_list = []
    for r, c in p.components:
        r_list.append(r)
        comp_list.append(c)

    # checks order of components
    assert isinstance(comp_list[0], PERM)
    assert isinstance(comp_list[1], BS)
    assert isinstance(comp_list[2], Circuit)
    assert isinstance(comp_list[3], PS)
    assert isinstance(comp_list[4], PERM)
    assert isinstance(comp_list[5], Circuit)
    assert isinstance(comp_list[6], PERM)

    # checks the position of elements
    assert r_list[0] == [4, 5, 6] # checks PERM added here to move extra mode out of the way
    assert r_list[1][0] == 0  # BS added at mode 0
    assert r_list[3][0] == 1  # checks PS at mode 1


def test_processor_add_detector():
    p = Processor("SLOS", 4)
    p.add(0, Detector.pnr())
    with pytest.raises(UnavailableModeException):
        p.add(0, PS(phi=0))  # Cannot add an optical component after a detector
    with pytest.raises(UnavailableModeException):
        p.add(0, Detector.pnr())  # Cannot add a detector after a detector


def test_remote_processor_creation(requests_mock):
    rp = RemoteProcessor(rpc_handler=get_rpc_handler(requests_mock), m=8)
    rp.add(0, BS())


def test_processor_composition_ports(requests_mock):
    ls = LogicalState([0, 0])
    cnot = catalog['postprocessed cnot'].build_processor()

    rp = RemoteProcessor(rpc_handler=get_rpc_handler(requests_mock), m=4)
    rp.min_detected_photons_filter(2)
    rp.add(0, cnot)
    rp.with_input(ls)

    # check that the input ports of the cnot are identical to the remote processor
    for mode in range(4):
        cnot_input_port = cnot.get_input_port(mode)
        rp_input_port = rp.get_input_port(mode)
        assert cnot_input_port == rp_input_port

        cnot_output_port = cnot.get_output_port(mode)
        rp_output_port = rp.get_output_port(mode)
        assert cnot_output_port == rp_output_port


def test_processor_building_feed_forward():
    m = 4
    u = Unitary(Matrix.random_unitary(m), "U0")
    p = Processor("SLOS", u)

    ffm = FFCircuitProvider(1, 0, Unitary(Matrix.random_unitary(1)), name="D2")

    for i in range(m):
        with pytest.raises(UnavailableModeException):
            p.add(i, ffm)  # Can't add because all modes are still photonic

    p.add(0, Detector.pnr())  # Mode 0 is now classical
    p.add(0, ffm)

    p.add(3, Detector.pnr())  # Mode 3 is now classical
    with pytest.raises(ValueError):
        p.add(3, ffm)  # Can't add because controlled circuit would be out of the processor

    with pytest.raises(ValueError):
        p.add(-1, ffm)  # Out of bound

    with pytest.raises(ValueError):
        p.add(4, ffm)  # Out of bound


def test_processor_feed_forward_multiple_layers():
    m = 4
    u = Unitary(Matrix.random_unitary(m), "U0")
    p = Processor("SLOS", u)
    p.add(2, Detector.pnr())
    mzi = catalog["mzi phase last"].build_circuit()
    mzi.name = "U1"
    ffc = FFConfigurator(1, -1, mzi, {"phi_a": 0, "phi_b": 0}, name="D1")
    p.add(2, ffc)
    p.add(0, Detector.pnr())
    p.add(1, Detector.pnr())

    ffc2 = FFConfigurator(2, 1, PS(phi=P("phi")), {"phi": 1.57}, name="D2")
    p.add(0, ffc2)

    expected = (Unitary, Barrier, Barrier, Detector, Barrier, FFConfigurator, Barrier, Detector, Detector, Barrier,
                FFConfigurator)
    for (r, c), expected_type in zip(p.components, expected):
        assert isinstance(c, expected_type)


def test_ff_controlled_circuit_size():
    m = 4
    u = Unitary(Matrix.random_unitary(m), "U0")
    p = Processor("SLOS", u)

    ffm = FFCircuitProvider(1, 0, Circuit(1), name="D2")
    ffm.add_configuration((1,), Circuit(2))  # Can add a larger circuit than the default one before it's used

    p.add(0, Detector.pnr())
    p.add(0, ffm)

    with pytest.raises(RuntimeError):
        ffm.add_configuration((1,), Circuit(3))
