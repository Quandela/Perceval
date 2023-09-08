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
from perceval import BasicState, generate_all_logical_states, SimulatorFactory
from perceval.components import catalog, BS, get_basic_state_from_ports
from perceval.algorithm import Analyzer


def test_fidelity_and_performance_compare_cnot():
    # Tests the performance and the fidelity of different CNOT in perceval
    # KLM CNOT
    klm_cnot = catalog["klm cnot"].build_processor()
    state_dict = {pcvl.components.get_basic_state_from_ports(klm_cnot._out_ports, state): str(
        state) for state in pcvl.utils.generate_all_logical_states(2)}
    analyzer_klm_cnot = Analyzer(klm_cnot, state_dict)
    analyzer_klm_cnot.compute(expected={"00": "00", "01": "01", "10": "11", "11": "10"})
    analyzer_klm_cnot_perf = pcvl.simple_float(analyzer_klm_cnot.performance)[1]

    assert pytest.approx(analyzer_klm_cnot.fidelity, 1E-4) == 1

    # Postprocessed CNOT
    postprocessed_cnot = catalog["postprocessed cnot"].build_processor()
    analyzer_postprocessed_cnot = Analyzer(postprocessed_cnot, state_dict)
    analyzer_postprocessed_cnot.compute(expected={"00": "00", "01": "01", "10": "11", "11": "10"})
    analyzer_postprocessed_cnot_perf = pcvl.simple_float(analyzer_postprocessed_cnot.performance)[1]

    assert analyzer_postprocessed_cnot.fidelity == 1

    # CNOT using CZ : called - Heralded CNOT
    heralded_cnot = catalog["heralded cnot"].build_processor()
    analyzer_heralded_cnot = Analyzer(heralded_cnot, state_dict)
    analyzer_heralded_cnot.compute(expected={"00": "00", "01": "01", "10": "11", "11": "10"})
    analyzer_heralded_cnot_perf = pcvl.simple_float(analyzer_heralded_cnot.performance)[1]

    assert pytest.approx(analyzer_heralded_cnot.fidelity) == 1

    assert analyzer_postprocessed_cnot_perf > analyzer_heralded_cnot_perf > analyzer_klm_cnot_perf


def check_cz_with_heralds(processor, herald_states, error=1E-6):
    """Check if the cz is correct

    :param processor: CZ processor to check, we assume that ctrl is on [0,1]
    and data on [2,3]
    :param herald_states: Basic state corresponding to the heralded or ancillary modes
    at the end of the circuit
    :param error: Tolerance for the real part of the prob_amplitude (when not expecting 0), defaults to 1E-6
    This param represent whether this cz gate is balanced
    """
    ports = [pcvl.Port(pcvl.Encoding.DUAL_RAIL, "")] * 2
    states = [get_basic_state_from_ports(ports, state) * herald_states for state in generate_all_logical_states(2)]
    ccz = processor
    sim = SimulatorFactory().build(ccz)
    value = None
    for i_state in states:
        for o_state in states:
            pa = sim.prob_amplitude(i_state, o_state)
            if i_state == o_state:
                if value is None:
                    value = np.real(pa)
                if i_state != BasicState("|0,1,0,1>") * herald_states:
                    assert pytest.approx(np.real(pa), error) == value
                else:
                    assert pytest.approx(np.real(pa), error) == -value
                assert pytest.approx(np.imag(pa)) == 0
            else:
                assert pytest.approx(np.imag(pa)) == 0
                assert pytest.approx(np.real(pa)) == 0


def test_cz_phase():
    check_cz_with_heralds(catalog["heralded cz"].build_processor(), BasicState("|1,1>"), 1E-3)

    processor = pcvl.Processor("SLOS", 4)
    processor.add(2, BS.H())
    processor.add(0, catalog["heralded cnot"].build_processor())
    processor.add(2, BS.H())
    check_cz_with_heralds(processor, BasicState("|1,1>"), 1E-3)

    processor = pcvl.Processor("SLOS", 4)
    processor.add(2, BS.H())
    processor.add(0, catalog["klm cnot"].build_processor())
    processor.add(2, BS.H())
    check_cz_with_heralds(processor, BasicState("|0,1,0,1>"), 1E-2)

    processor = pcvl.Processor("SLOS", 4)
    processor.add(2, BS.H())
    processor.add(0, catalog["postprocessed cnot"].build_processor())
    processor.add(2, BS.H())
    check_cz_with_heralds(processor, BasicState("|0,0>"))


@pytest.mark.skip(reason="redundant with overhead test")
@pytest.mark.parametrize("cnot_gate", ["klm cnot", "postprocessed cnot", "heralded cnot"])
def test_inverse_cnot(cnot_gate):
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
    processor = pcvl.Processor("SLOS", 4)
    processor.add([0, 1], BS.H())
    processor.add([2, 3], BS.H())
    # Commented lines are use to compare with a26b0bd (0.8.1 before cnot fix)
    # processor.add([2, 3, 0, 1], catalog["postprocessed cnot"].as_processor().build()) # < 0.9.0
    # processor.clear_postprocess() # < 0.9.0
    processor.add([2, 3, 0, 1], catalog[cnot_gate].build_processor())  # >= 0.9.0
    processor.add([0, 1], BS.H())
    processor.add([2, 3], BS.H())
    # processor.set_postprocess(lambda o: (o[0] + o[1] == 1) and (o[2] + o[3] == 1)) # < 0.9.0

    # state_dict = {BasicState("|1,0,1,0>"): '00', BasicState("|1,0,0,1>"): '01', BasicState("|0,1,1,0>"): '10', BasicState("|0,1,0,1>"): '11'} # < 0.9.0
    state_dict = {pcvl.components.get_basic_state_from_ports(processor._out_ports, state): str(
        state) for state in pcvl.utils.generate_all_logical_states(2)}  # >= 0.9.0
    analyzer = Analyzer(processor, state_dict)
    analyzer.compute(expected={"00": "00", "01": "01", "10": "11", "11": "10"})

    if cnot_gate == "klm cnot":
        assert pytest.approx(analyzer.fidelity, 1E-4) == 1
    elif cnot_gate == "heralded cnot":
        assert pytest.approx(analyzer.fidelity) == 1
    else:
        assert analyzer.fidelity == 1
