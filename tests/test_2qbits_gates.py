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

import perceval as pcvl
from perceval import BasicState, generate_all_logical_states, SimulatorFactory, Processor
from perceval.components import catalog, BS, get_basic_state_from_ports
from perceval.algorithm import Analyzer


def test_fidelity_and_performance_cnot():
    # Tests the performance and the fidelity of different CNOT in perceval
    ports = [pcvl.Port(pcvl.Encoding.DUAL_RAIL, "")] * 2
    state_dict = {pcvl.components.get_basic_state_from_ports(ports, state): str(
        state) for state in pcvl.utils.generate_all_logical_states(2)}
    expected = {"00": "00", "01": "01", "10": "11", "11": "10"}

    # KLM CNOT
    analyzer_klm_cnot = Analyzer(
        catalog["klm cnot"].build_processor(), state_dict)
    analyzer_klm_cnot.compute(expected=expected)

    assert pytest.approx(analyzer_klm_cnot.fidelity) == 1

    # Postprocessed CNOT
    analyzer_postprocessed_cnot = Analyzer(
        catalog["postprocessed cnot"].build_processor(), state_dict)
    analyzer_postprocessed_cnot.compute(expected=expected)

    assert pytest.approx(analyzer_postprocessed_cnot.fidelity) == 1

    # CNOT using CZ : called - Heralded CNOT
    analyzer_heralded_cnot = Analyzer(
        catalog["heralded cnot"].build_processor(), state_dict)
    analyzer_heralded_cnot.compute(expected=expected)

    assert pytest.approx(analyzer_heralded_cnot.fidelity) == 1

    assert analyzer_postprocessed_cnot.performance > analyzer_heralded_cnot.performance > analyzer_klm_cnot.performance


def check_cz_with_heralds_or_ancillaries(processor, herald_states, error=1E-6):
    """Check if the cz is correct

    Meaning checking that the CZ gate probability amplitude matrix should be:
    ⎡r*exp(iθ)  0           0           0             ⎤
    ⎢0          r*exp(iθ)   0           0             ⎥
    ⎢0          0           r*exp(iθ)   0             ⎥
    ⎣0          0           0           r*exp(i(θ+φ)) ⎦
    With r the modulus (r!=0), θ the global phase and φ the rotation angle of our gate (for CZ is φ=π)

    :param processor: CZ processor to check, we assume that ctrl is on [0,1]
    and data on [2,3]
    :param herald_states: Basic state corresponding to the heralded or ancillary modes
    at the end of the circuit
    :param error: Tolerance for the modulus of the prob_amplitude (when not expecting 0), defaults to 1E-6
    This param represent whether this cz gate is balanced
    """
    ports = [pcvl.Port(pcvl.Encoding.DUAL_RAIL, "")] * 2
    states = [get_basic_state_from_ports(
        ports, state) * herald_states for state in generate_all_logical_states(2)]
    sim = SimulatorFactory().build(processor)
    data_state = BasicState("|0,1,0,1>") * herald_states
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
                    assert pytest.approx(phase) == phase_value
                else:
                    assert pytest.approx(phase) == phase_value + cm.pi

            else:
                assert pytest.approx(modulus) == 0
    return modulus_value


def test_cz_and_cnot_phases_and_modulus():
    # Testing phases and modulus of CCZ
    check_cz_with_heralds_or_ancillaries(
        catalog["heralded cz"].build_processor(), BasicState("|1,1>"))

    # Testing phases and modulus of heralded cnot by transforming it in a CZ gate with Hadamard gates
    processor = Processor("SLOS", 4)
    processor.add(2, BS.H())
    processor.add(0, catalog["heralded cnot"].build_processor())
    processor.add(2, BS.H())
    check_cz_with_heralds_or_ancillaries(processor, BasicState("|1,1>"))
    # Testing phases and modulus of klm cnot by transforming it in a CZ gate with Hadamard gates
    processor = Processor("SLOS", 4)
    processor.add(2, BS.H())
    processor.add(0, catalog["klm cnot"].build_processor())
    processor.add(2, BS.H())
    check_cz_with_heralds_or_ancillaries(processor, BasicState("|0,1,0,1>"))

    # Testing phases and modulus of postprocessed cnot by transforming it in a CZ gate with Hadamard gates
    processor = Processor("SLOS", 4)
    processor.add(2, BS.H())
    processor.add(0, catalog["postprocessed cnot"].build_processor())
    processor.add(2, BS.H())
    check_cz_with_heralds_or_ancillaries(processor, BasicState("|0,0>"))


@pytest.mark.skip(reason="redundant with overhead test")
@pytest.mark.parametrize("cnot_gate", ["klm cnot", "postprocessed cnot", "heralded cnot"])
def test_inverted_cnot(cnot_gate):
    """Test cnot gate phase by inverting it with Hadamard gates

     ╭───╮   ╭──────────╮   ╭───╮             ╭──────────╮
 ────┤ H ├───┤  DATA    ├───┤ H ├────      ───┤  CTRL    ├───
 ────┤   ├───┤          ├───┤   ├────      ───┤          ├───
     ╰───╯   │  CNOT    │   ╰───╯      <=>    │  CNOT    │
     ╭───╮   │          │   ╭───╮             │          │
 ────┤ H ├───┤  CTRL    ├───┤ H ├────      ───┤  DATA    ├───
 ────┤   ├───┤          ├───┤   ├────      ───┤          ├───
     ╰───╯   ╰──────────╯   ╰───╯             ╰──────────╯

    :param cnot_gate: cnot catalog gate
    """
    processor = Processor("SLOS", 4)
    processor.add([0, 1], BS.H())
    processor.add([2, 3], BS.H())
    # Commented lines are use to compare with a26b0bd (0.8.1 before cnot fix)
    # processor.add([2, 3, 0, 1], catalog["postprocessed cnot"].as_processor().build()) # < 0.9.0
    # processor.clear_postprocess() # < 0.9.0
    processor.add(
        [2, 3, 0, 1], catalog[cnot_gate].build_processor())  # >= 0.9.0
    processor.add([0, 1], BS.H())
    processor.add([2, 3], BS.H())

    state_dict = {pcvl.components.get_basic_state_from_ports(processor._out_ports, state): str(
        state) for state in pcvl.utils.generate_all_logical_states(2)}
    analyzer = Analyzer(processor, state_dict)
    analyzer.compute(expected={"00": "00", "01": "01", "10": "11", "11": "10"})

    if cnot_gate == "klm cnot":
        assert pytest.approx(analyzer.fidelity, 1E-4) == 1
    elif cnot_gate == "heralded cnot":
        assert pytest.approx(analyzer.fidelity) == 1
    else:
        assert analyzer.fidelity == 1
