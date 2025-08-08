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

from __future__ import annotations
from multipledispatch import dispatch

from collections import Counter

import random
import numpy as np

from .states import BSDistribution, BSCount, BSSamples, SVDistribution, BasicState, StateVector, NoisyFockState, \
    FockState


def _deduce_count(**kwargs) -> int:
    count = kwargs.get("count")
    if count is not None:
        return count
    max_shots = kwargs.get("max_shots")
    max_samples = kwargs.get("max_samples")
    if max_shots is not None and max_samples is not None:
        return min(max_samples, max_shots)  # Not accurate in terms of shot limit
    count = max_shots or max_samples
    if count is None:
        raise RuntimeError("kwargs does not contain sample count information")
    return count


# Conversion functions (samples <=> probs <=> sample_count)
def samples_to_sample_count(sample_list: list[NoisyFockState] | list[FockState]) -> Counter[NoisyFockState] | Counter[FockState]:
    """
    Convert a chronological measured sample list to a state count

    :param sample_list: the list to convert
    :return: the state count
    """
    return Counter(sample_list)


def samples_to_probs(sample_list: list[NoisyFockState] | list[FockState]) -> SVDistribution:
    """
    Convert a chronological measured sample list to a state distribution

    :param sample_list: the list to convert
    :return: the state distribution
    """
    return sample_count_to_probs(samples_to_sample_count(sample_list))


def probs_to_sample_count(probs: BSDistribution, **kwargs) -> BSCount:
    """
    Convert a measured state probability distribution to a state count.

    This conversion artificially adds random sampling noise, following a normal law, to the result.

    :param probs: the distribution to convert

    :keyword count:
        (``int``) -- The final number of samples to generate. Can be None if either of the remaining kwargs is defined.
    :keyword max_shots:
        (``int``) -- If both ``max_shots`` and ``max_samples`` are given, then the minimum of the two will be used.
        Else, the one defined will be used if ``count`` is not given.
    :keyword max_samples:
        (``int``) -- See ``max_shots``.

    :return: the state count
    """
    count = _deduce_count(**kwargs)
    if count < 1:
        return BSCount()
    perturbed_dist = {state: max(prob + np.random.normal(scale=(prob * (1 - prob) / count) ** .5), 0)
                      for state, prob in probs.items()}
    prob_sum = sum(perturbed_dist.values())
    if prob_sum == 0:
        return samples_to_sample_count(probs_to_samples(probs, count=count))
    fac = 1 / prob_sum
    perturbed_dist = {key: fac * prob for key, prob in perturbed_dist.items()}  # Renormalisation
    if max(perturbed_dist.values()) * count < 1:
        return samples_to_sample_count(probs_to_samples(probs, count=count))

    results = BSCount()
    for state in perturbed_dist:
        results.add(state, round(perturbed_dist[state] * count))
    # Artificially deal with the rounding errors
    diff = round(count - sum(results.values()))
    if diff > 0:
        results[random.choice(list(results.keys()))] += diff
    elif diff < 0:
        while diff < 0:
            k = random.choice(list(results.keys()))
            current_diff = max(-results[k], diff)
            diff -= current_diff
            results[k] += current_diff
    return results


def probs_to_samples(probs: BSDistribution, **kwargs) -> BSSamples:
    """
    Convert a measured state probability distribution to a chronological list of  samples

    :param probs: the distribution to convert

    :keyword count:
        (``int``) -- The final number of samples to generate. Can be None if either of the remaining kwargs is defined.
    :keyword max_shots:
        (``int``) -- If both ``max_shots`` and ``max_samples`` are given, then the minimum of the two will be used.
        Else, the one defined will be used if ``count`` is not given.
    :keyword max_samples:
        (``int``) -- See ``max_shots``.

    :return: the sample list
    """
    count = _deduce_count(**kwargs)
    return probs.sample(count)


@dispatch(BSCount)
def sample_count_to_probs(sample_count: BSCount) -> BSDistribution:
    """
    Convert a state count to a state probability distribution

    :param sample_count: the state count
    :return: the state probability distribution
    """
    bsd = BSDistribution()
    for state, count in sample_count.items():
        if count == 0:
            continue
        if count < 0:
            raise RuntimeError(f"A sample count must be positive (got {count})")
        bsd[state] = count
    if len(bsd):
        bsd.normalize()
    return bsd


@dispatch(Counter)
def sample_count_to_probs(sample_count: Counter[NoisyFockState] | Counter[FockState]) -> SVDistribution:
    """
    Convert a state count to a state probability distribution

    :param sample_count: the state count
    :return: the state probability distribution
    """
    svd = SVDistribution()
    for state, count in sample_count.items():
        if count == 0:
            continue
        if count < 0:
            raise RuntimeError(f"A sample count must be positive (got {count})")
        svd[StateVector(BasicState(state))] = count
    if len(svd):
        svd.normalize()
    return svd


def sample_count_to_samples(sample_count: BSCount, **kwargs) -> BSSamples:
    """
    Convert a state count to a chronological list of samples, by randomly sampling on the count

    :param sample_count: the state count

    :keyword count:
        (``int``) -- The final number of samples to generate. Can be None to deduce it from the number of samples in the BSCount.
    :keyword max_shots:
        (``int``) -- If both ``max_shots`` and ``max_samples`` are given, then the minimum of the two will be used.
        Else, the one defined will be used if ``count`` is not given.
    :keyword max_samples:
        (``int``) -- See ``max_shots``.

    :return: the sample list
    """
    try:
        count = _deduce_count(**kwargs)
    except RuntimeError:
        count = sum(sample_count.values())
    return sample_count_to_probs(sample_count).sample(count)
