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

from .statevector import BasicState, SVDistribution

from typing import Dict, List
import numpy as np


# Conversion functions (samples <=> probs <=> sample_count)
def samples_to_sample_count(sample_list: List[BasicState]) -> Dict[BasicState, int]:
    results = {}
    for s in sample_list:
        if s not in results:
            results[s] = sample_list.count(s)
    return results


def samples_to_probs(sample_list: List[BasicState]) -> SVDistribution:
    return sample_count_to_probs(samples_to_sample_count(sample_list))


def probs_to_sample_count(probs: SVDistribution, count: int) -> Dict[BasicState, int]:
    perturbed_dist = {state: max(prob + np.random.normal(scale=(prob * (1 - prob) / count) ** .5), 0)
                      for state, prob in probs.items()}
    fac = 1 / sum(prob for prob in perturbed_dist.values())
    perturbed_dist = {key: fac * prob for key, prob in perturbed_dist.items()}  # Renormalisation
    results = dict()
    for state in perturbed_dist:
        results[state] = int(np.round(perturbed_dist[state] * count))
    return results


def probs_to_samples(probs: SVDistribution, count: int) -> List[BasicState]:
    return probs.sample(count)


def sample_count_to_probs(sample_count: Dict[BasicState, int]):
    svd = SVDistribution()
    n_samples = 0
    for state, count in sample_count.items():
        if count == 0:
            continue
        if count < 0:
            raise RuntimeError(f"A sample count must be positive (got {count})")
        svd[state] = count
        n_samples += count
    for state, value in svd.items():
        svd[state] = value / n_samples
    return svd


def sample_count_to_samples(sample_count: Dict[BasicState, int], count: int):
    return sample_count_to_probs(sample_count).sample(count)
