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

import argparse
from dataclasses import dataclass
import perceval
import scipy
import warnings
from perceval.backends import BackendFactory
from perceval.components.linear_circuit import Circuit
from perceval.components.unitary_components import BS
from perceval.utils.states import BSDistribution, BasicState


@dataclass
class Problem:
    circuit: Circuit
    input_state: BasicState
    expected_distribution: BSDistribution

def problem_1() -> Problem:
    return Problem(
        circuit = Circuit(2) // BS.H(),
        input_state = BasicState([1, 1]),
        expected_distribution = BSDistribution({ BasicState([2, 0]): 0.5, BasicState([0, 2]): 0.5 })
    )


def problem_2() -> Problem:
    return Problem(
        circuit = Circuit(4) // BS.H() // (2, BS.H()),
        input_state = BasicState([1, 0, 1, 0]),
        expected_distribution = BSDistribution({ BasicState([1, 0, 1, 0]): 0.25, BasicState([1, 0, 0, 1]): 0.25, BasicState([0, 1, 1, 0]): 0.25, BasicState([0, 1, 0, 1]): 0.25 })
    )

parser = argparse.ArgumentParser(description='Runs chi-squared test on Clifford samplings')
parser.add_argument('--samplecount', '-s',
                    type=int, action='store', default = 1e6,
                    help='maximum number of samples to compute')
args = parser.parse_args()

backend = perceval.backends.BackendFactory.get_backend("CliffordClifford2017")
problem = problem_1()
backend.set_circuit(problem.circuit)
backend.set_input_state(problem.input_state)
expected = problem.expected_distribution


print(f"{'samples':<12} {'distance':<10} (p-value)")
n = 100
while n <= args.samplecount:
    bsc = perceval.utils.samples_to_sample_count(backend.samples(n))
    minimum_sample_count = min(bsc.values())
    if minimum_sample_count < 5:
        warnings.warn("Too few samples, seen notes in https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chisquare.html")
    dist = perceval.utils.sample_count_to_probs(bsc)
    observed = [dist[output_state] for output_state in expected.keys()]

    tvd = perceval.utils.tvd_dist(expected, dist)

    chi = scipy.stats.chisquare(observed)
    if chi.pvalue < 0.01:
        warnings.warn("(A p-value below 1% tends to think that the samples aren't distributed as expected)")

    print(f"{n:<12} {tvd:.8f} ({chi.pvalue:.5f})")

    n *= 10
