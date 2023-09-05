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
from math import pi, sqrt

import perceval as pcvl
from perceval import PostSelect, Matrix, BasicState
from perceval.algorithm import Analyzer, Sampler
from perceval.utils import Encoding, generate_all_logical_states
from perceval.components import catalog, BS, PERM, Unitary, PS, Port, get_basic_state_from_ports


def test_toffoli_gate_fidelity():
    t = catalog['toffoli'].build_processor()
    state_dict = {get_basic_state_from_ports(t._out_ports, state): str(
        state) for state in generate_all_logical_states(3)}
    ca = Analyzer(t, input_states=state_dict)
    ca.compute(expected={"000": "000", "001": "001", "010": "010", "011": "011",
               "100": "100", "101": "101", "110": "111", "111": "110"})
    assert ca.fidelity == 1


@pytest.mark.skip(reason="Work in progress")
def test_toffoli_phase():
    """Heralding ctrl one after the other to do the same test as test_inverse_cnot_with_H
    in tests/test_2qbits_gates.py
    """
    cnot01 = pcvl.Processor("SLOS", 12)
    toffoli_port_mapping = [4,5,2,3,0,1]
    # toffoli_port_mapping = [5,4,3,2,1,0]
    toffoli_port_mapping.extend([i for i in range(6,12)])
    cnot01.add([0, 1], BS.H()) \
        .add([4, 5], BS.H()) \
        .add(toffoli_port_mapping, catalog["toffoli"].build_circuit()) \
        .add([0, 1], BS.H()) \
        .add([4, 5], BS.H()) \
        .add_port(0, Port(Encoding.DUAL_RAIL, 'ctrl')) \
        .add_herald(2,0) \
        .add_herald(3,0) \
        .add_port(4, Port(Encoding.DUAL_RAIL, 'data')) \
        .add_herald(6, 0) \
        .add_herald(7, 0) \
        .add_herald(8, 0) \
        .add_herald(9, 0) \
        .add_herald(10, 0) \
        .add_herald(11, 0)
    cnot01.set_postselection(pcvl.PostSelect("[0,1]==1 & [4,5]==1"))

    state_dict = {get_basic_state_from_ports(cnot01._out_ports, state): str(
        state) for state in generate_all_logical_states(2)}
    a_cnot01 = Analyzer(cnot01, input_states=state_dict)
    a_cnot01.compute()
    pcvl.pdisplay(a_cnot01)
    # a_cnot01.compute(expected={"00": "00", "01": "01", "10": "11", "11": "10"})
    # assert a_cnot01.fidelity == 1

    cnot02 = pcvl.Processor("SLOS", 12)
    cnot02.add([2, 3], BS.H()) \
        .add([4, 5], BS.H()) \
        .add(toffoli_port_mapping, catalog["toffoli"].build_circuit()) \
        .add([2, 3], BS.H()) \
        .add([4, 5], BS.H()) \
        .add_herald(0,0) \
        .add_herald(1,0) \
        .add_port(2, Port(Encoding.DUAL_RAIL, 'ctrl')) \
        .add_port(4, Port(Encoding.DUAL_RAIL, 'data')) \
        .add_herald(6, 0) \
        .add_herald(7, 0) \
        .add_herald(8, 0) \
        .add_herald(9, 0) \
        .add_herald(10, 0) \
        .add_herald(11, 0)
    cnot02.set_postselection(pcvl.PostSelect("[2,3]==1 & [4,5]==1"))
    a_cnot02 = Analyzer(cnot02, input_states=state_dict)
    pcvl.pdisplay(a_cnot02)
    # a_cnot02.compute(expected={"00": "00", "01": "01", "10": "11", "11": "10"})
    # assert a_cnot02.fidelity == 1


def test_full_toffoli():
    a = 1 / sqrt(2)
    H3 = Matrix([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, a, 0, 0, 0, 0, 0, 0, 0, a, 0],
                 [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                 [0, 0, a, 0, 0, 0, 0, 0, 0, 0, -a, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
    Z3 = Matrix([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
    # U is obtained by a method not communicated
    U = Matrix([[0.509824528533959, 0.321169327626332 + 0.556281593281541j, 0, 0.330393705586394, -0.165196852793197 - 0.286129342288294j, -0.165196852793197 + 0.286129342288294j, 0, 0, 0, 0, 0, 0],
                [0, 0.509824528533959, 0.321169327626332 + 0.556281593281541j, -0.165196852793197
                 + 0.286129342288294j, 0.330393705586394, -0.165196852793197 - 0.286129342288294j, 0, 0, 0, 0, 0, 0],
                [0.321169327626332 + 0.556281593281541j, 0, 0.509824528533959, -0.165196852793197
                 - 0.286129342288294j, -0.165196852793197 + 0.286129342288294j, 0.330393705586394, 0, 0, 0, 0, 0, 0],
                [0.330393705586394, -0.165196852793197 - 0.286129342288294j, -0.165196852793197
                 + 0.286129342288294j, -0.509824528533959, 0, -0.321169327626332 + 0.556281593281541j, 0, 0, 0, 0, 0, 0],
                [-0.165196852793197 + 0.286129342288294j, 0.330393705586394, -0.165196852793197
                 - 0.286129342288294j, -0.321169327626332 + 0.556281593281541j, -0.509824528533959, 0, 0, 0, 0, 0, 0, 0],
                [-0.165196852793197 - 0.286129342288294j, -0.165196852793197 + 0.286129342288294j, 0.330393705586394,
                 0, -0.321169327626332 + 0.556281593281541j, -0.509824528533959, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0.509824528533959, 0.860278414296864, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0.860278414296864, -0.509824528533959, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0.509824528533959, 0.860278414296864, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0.860278414296864, -0.509824528533959, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.509824528533959, 0.860278414296864],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.860278414296864, -0.509824528533959]])
    Pin = PERM([6, 0, 8, 1, 10, 2, 3, 4, 5, 7, 9, 11])
    Pout = PERM([1, 3, 5, 6, 7, 8, 0, 9, 2, 10, 4, 11])
    toffoli = Pin // Unitary(Z3 @ H3 @ U @ H3 @ Z3) // Pout
    toffoli.add(2, PS(pi))
    toffoli.add(0, PS(pi / 2))

    processor = pcvl.Processor("SLOS", toffoli)
    processor.add_port(0, Port(Encoding.DUAL_RAIL, 'ctrl0')) \
        .add_port(2, Port(Encoding.DUAL_RAIL, 'ctrl1')) \
        .add_port(4, Port(Encoding.DUAL_RAIL, 'data')) \
        .add_herald(6, 0) \
        .add_herald(7, 0) \
        .add_herald(8, 0) \
        .add_herald(9, 0) \
        .add_herald(10, 0) \
        .add_herald(11, 0)
    processor.set_postselection(pcvl.PostSelect("[0,1]==1 & [2,3]==1 & [4,5]==1"))

    state_dict = {get_basic_state_from_ports(processor._out_ports, state): str(
        state) for state in generate_all_logical_states(3)}
    ca = Analyzer(processor, input_states=state_dict)
    ca.compute(expected={"000": "000", "001": "001", "010": "010", "011": "011",
               "100": "100", "101": "101", "110": "111", "111": "110"})
    assert ca.fidelity == 1
