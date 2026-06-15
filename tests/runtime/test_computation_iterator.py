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

from perceval import Experiment, FockState, Computation, BSSamples, NoiseModel, Command
from perceval.runtime.computation_iterator import ComputationIterator
from tests._test_utils import assert_experiment_equals


probs_command = Command("samples", [("max_samples", int, False)])

def test_iterator_extension():
    e = Experiment(2)
    e.with_input(FockState([1, 0]))

    base_comp = Computation(probs_command, e)
    comp = ComputationIterator(base_comp)

    comp.add_iteration(max_samples=10000)
    comp.add_iteration(max_samples=50000)

    sub_comps = list(comp)

    assert len(sub_comps) == 2

    assert_experiment_equals(sub_comps[0].experiment, e)
    assert_experiment_equals(sub_comps[1].experiment, e)

    assert sub_comps[0].parameters["max_samples"] == 10000
    assert sub_comps[1].parameters["max_samples"] == 50000


def test_iteration_parsing():
    e = Experiment(2)

    base_comp = Computation(probs_command, e)
    comp = ComputationIterator(base_comp)

    comp.add_iteration(max_samples=10000)
    comp.add_iteration(max_samples=50000)

    parsed = {}
    inserter = comp.make_inserter(parsed)

    fake_results = [{"results": BSSamples([FockState([1, 0])])}, {"results": BSSamples([FockState([0, 1])])}]
    for res in fake_results:
        inserter(res)

    assert "results_list" in parsed
    assert len(parsed["results_list"]) == 2
    assert parsed["results_list"] == fake_results

    assert "iteration" in parsed["results_list"][0]
    assert parsed["results_list"][0]["iteration"] == {"max_samples": 10000}
    assert parsed["results_list"][1]["iteration"] == {"max_samples": 50000}
