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

import perceval as pcvl
from perceval.components import catalog
from perceval.algorithm import Analyzer

STATES = {
    pcvl.BasicState([1, 0, 1, 0]): "00",
    pcvl.BasicState([1, 0, 0, 1]): "01",
    pcvl.BasicState([0, 1, 1, 0]): "10",
    pcvl.BasicState([0, 1, 0, 1]): "11"
}


def test_performance_compare_cnot():
    # Tests the performance of different CNOT in perceval
    # KLM CNOT
    klm_cnot = catalog["klm cnot"].build_processor()
    analyzer_klm_cnot = Analyzer(klm_cnot, STATES)
    analyzer_klm_cnot.compute(expected={"00": "00", "01": "01", "10": "11", "11": "10"})
    analyzer_klm_cnot_perf = pcvl.simple_float(analyzer_klm_cnot.performance)[1]

    # Postprocessed CNOT
    postprocessed_cnot = catalog["postprocessed cnot"].build_processor()
    analyzer_postprocessed_cnot = Analyzer(postprocessed_cnot, STATES)
    analyzer_postprocessed_cnot.compute(expected={"00": "00", "01": "01", "10": "11", "11": "10"})
    analyzer_postprocessed_cnot_perf = pcvl.simple_float(analyzer_postprocessed_cnot.performance)[1]

    # CNOT using CZ : called - Heralded CNOT
    heralded_cnot = catalog["heralded cnot"].build_processor()
    analyzer_heralded_cnot = Analyzer(heralded_cnot, STATES)
    analyzer_heralded_cnot.compute(expected={"00": "00", "01": "01", "10": "11", "11": "10"})
    analyzer_heralded_cnot_perf = pcvl.simple_float(analyzer_heralded_cnot.performance)[1]

    assert analyzer_postprocessed_cnot_perf > analyzer_heralded_cnot_perf > analyzer_klm_cnot_perf
