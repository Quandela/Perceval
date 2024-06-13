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
import cmath as cm

from perceval import BasicState, Port, Processor, Encoding, PostSelect
from perceval.algorithm import Analyzer
from perceval.utils import generate_all_logical_states
from perceval.simulators import SimulatorFactory
from perceval.components import catalog, BS, get_basic_state_from_ports


def check_controlled_gates_and_get_performance(n, alpha, processor, herald_states, error=1E-6):
    """Check if the CC..Z(alpha) is correct

    Meaning checking that the CC...Z(alpha) gate probability amplitude matrix should be: 
                        Id_(2^n) +(exp(i.alpha)-1) |1...1><1...1|

    :param processor: CCZ processor to check, we assume that ctrl is on [0,1] and [2,3]
    and data on [4,5]
    :param herald_states: Basic state corresponding to the heralded or ancillary modes
    at the end of the circuit
    :param error: Tolerance for the modulus of the prob_amplitude (when not expecting 0), defaults to 1E-6
    This param represent whether this CCZ gate is balanced
    """
    ports = [Port(Encoding.DUAL_RAIL, "")] * n
    states = [get_basic_state_from_ports(
        ports, state) * herald_states for state in generate_all_logical_states(n)]
    sim = SimulatorFactory().build(processor)

    data_state = BasicState("|0,1"+(n-1)*",0,1"+">") * herald_states
    modulus_value = None
    phase_value = None
    for i_state in states:
        for o_state in states:
            pa = sim.prob_amplitude(i_state, o_state)
            modulus = abs(pa)
            phase = cm.phase(pa)
            if i_state == o_state:
                if modulus_value is None:
                    modulus_value = modulus
                assert pytest.approx(modulus, error) == modulus_value
                assert modulus != 0

                if i_state != data_state:
                    if phase_value is None:
                        phase_value = phase
                    assert pytest.approx(phase) == phase_value % (2 * cm.pi)
                else:
                    delta_phase = (phase - (phase_value + alpha) +
                                   cm.pi) % (2*cm.pi) - cm.pi
                    assert pytest.approx(delta_phase) == 0

            else:
                assert pytest.approx(modulus) == 0
    return modulus_value


def test_controlled_gates():
    for n in [2, 3, 4]:
        for alpha in [cm.pi, cm.pi/3, -cm.pi/4, 1.]:

            # Testing phases and modulus of CCZ
            check_controlled_gates_and_get_performance(n, alpha,
                                                       catalog['postprocessed controlled gate'].build_processor(n=n, alpha=alpha), BasicState("|0"+(2*n-1)*",0"+">"))


def test_ccz_and_toffoli_phases_and_modulus():
    # Testing phases and modulus of CCZ
    modulus_ccz = check_controlled_gates_and_get_performance(3,
                                                             cm.pi,
                                                             catalog['postprocessed ccz'].build_processor(), BasicState("|0,0,0,0,0,0>"))

    # Testing phases and modulus of Toffoli by transforming it in a CCZ gate with Hadamard gates
    ccz = Processor("SLOS", 6)
    ccz.add(4, BS.H()) \
        .add(0, catalog["toffoli"].build_processor()) \
        .add(4, BS.H())
    modulus_ccz_with_toffoli = check_controlled_gates_and_get_performance(3,
                                                                          cm.pi,
                                                                          catalog['postprocessed ccz'].build_processor(), BasicState("|0,0,0,0,0,0>"))
    assert modulus_ccz == pytest.approx(modulus_ccz_with_toffoli)

    # Testing truth table of Toffoli (redundant with above test)
    toffoli = catalog['toffoli'].build_processor()
    state_dict = {get_basic_state_from_ports(toffoli._out_ports, state): str(
        state) for state in generate_all_logical_states(3)}
    a_toffoli = Analyzer(toffoli, input_states=state_dict)
    a_toffoli.compute(expected={"000": "000", "001": "001", "010": "010", "011": "011",
                                "100": "100", "101": "101", "110": "111", "111": "110"})
    assert a_toffoli.fidelity == 1
    # Checking that Toffoli performance is the modulus**2 of the CCZ gate (since Toffoli comes from CCZ gate)
    assert a_toffoli.performance == pytest.approx(modulus_ccz**2)


@pytest.mark.skip(reason="redundant with overhead test")
def test_inverted_sub_cnot():
    """Heralding ctrl one after the other to do the same test as test_inverted_cnot
    in tests/test_2qbits_gates.py
    """
    toffoli = catalog["toffoli"].build_processor()
    toffoli.remove_port(2) \
        .add_herald(2, 0)\
        .add_herald(3, 1)
    toffoli.clear_postselection()
    toffoli.set_postselection(PostSelect("[0,1]==1 & [4,5]==1"))
    cnot0 = Processor("SLOS", 4)
    cnot0.add([0, 1], BS.H()) \
        .add([2, 3], BS.H()) \
        .add([2, 3, 0, 1], toffoli) \
        .add([0, 1], BS.H()) \
        .add([2, 3], BS.H())

    state_dict = {get_basic_state_from_ports(cnot0._out_ports, state): str(
        state) for state in generate_all_logical_states(2)}
    a_cnot0 = Analyzer(cnot0, input_states=state_dict)
    a_cnot0.compute(expected={"00": "00", "01": "01", "10": "11", "11": "10"})
    assert a_cnot0.fidelity == 1

    toffoli = catalog["toffoli"].build_processor()
    toffoli.remove_port(0) \
        .add_herald(0, 0)\
        .add_herald(1, 1)
    toffoli.clear_postselection()
    toffoli.set_postselection(PostSelect("[2,3]==1 & [4,5]==1"))
    cnot1 = Processor("SLOS", 4)
    cnot1.add([0, 1], BS.H()) \
        .add([2, 3], BS.H()) \
        .add([2, 3, 0, 1], toffoli) \
        .add([0, 1], BS.H()) \
        .add([2, 3], BS.H())
    a_cnot1 = Analyzer(cnot1, input_states=state_dict)
    a_cnot1.compute(expected={"00": "00", "01": "01", "10": "11", "11": "10"})
    assert a_cnot1.fidelity == 1
