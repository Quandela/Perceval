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

import perceval as pcvl
from perceval import BasicState
from perceval.components import catalog, BS, PERM
from perceval.algorithm import Analyzer, Sampler


def test_fidelity_and_performance_compare_cnot():
    # Tests the performance and the fidelity of different CNOT in perceval
    # KLM CNOT
    klm_cnot = catalog["klm cnot"].build_processor()
    state_dict = {pcvl.components.get_basic_state_from_ports(klm_cnot._out_ports, state): str(
        state) for state in pcvl.utils.generate_all_logical_states(2)}
    analyzer_klm_cnot = Analyzer(klm_cnot, state_dict)
    analyzer_klm_cnot.compute(expected={"00": "00", "01": "01", "10": "11", "11": "10"})
    analyzer_klm_cnot_perf = pcvl.simple_float(analyzer_klm_cnot.performance)[1]

    assert pytest.approx(analyzer_klm_cnot.fidelity, 10E-5) == 1

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


# We could use H gates on both size of a CZ gate to check CZ phase
# but since our heralded CNOT is built this way, we don't need to
@pytest.mark.parametrize("cnot_gate", ["klm cnot", "postprocessed cnot", "heralded cnot"])
def test_cnot_phase(cnot_gate):
    processor = pcvl.Processor("SLOS", 4)
    processor.add([2, 3], PERM([1, 0]))
    processor.add([0, 1], BS.H())
    processor.add([2, 3], BS.H())
    # Commented lines are use to compare with a26b0bd (0.8.1 before cnot fix)
    # processor.add([0, 1, 2, 3], catalog["postprocessed cnot"].as_processor().build()) # < 0.9.0
    # processor.clear_postprocess() # < 0.9.0
    processor.add([0, 1, 2, 3], catalog[cnot_gate].build_processor())  # >= 0.9.0
    processor.add([0, 1], BS.H())
    processor.add([2, 3], BS.H())
    # processor.set_postprocess(lambda o: (o[0] + o[1] == 1) and (o[2] + o[3] == 1)) # < 0.9.0

    processor.with_input(BasicState([1, 0, 1, 0]))
    sampler = Sampler(processor)
    nb_sample = 10
    samples = sampler.samples(nb_sample)
    assert samples['results'].count(BasicState([0, 1, 0, 1])) == nb_sample
