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

import random

from abc import ABC, abstractmethod
from collections import defaultdict
from copy import copy
from multipledispatch import dispatch
try:
    from typing import TypeAlias
except ImportError:
    from typing_extensions import TypeAlias  # Only used with python 3.9

import exqalibur as xq
from perceval.utils.logging import get_logger, channel

from .globals import global_params
from .qmath import exponentiation_by_squaring

BasicState: TypeAlias = xq.FockState
BSCount: TypeAlias = xq.BSCount
BSSamples: TypeAlias = xq.BSSamples
StateVector: TypeAlias = xq.StateVector


def allstate_iterator(input_state: BasicState | StateVector, mask: xq.FSMask = None) -> BasicState:
    """Iterator on all possible output states compatible with mask generating StateVector

    :param input_state: a given input state vector
    :param mask: an optional mask
    :return: list of output_state
    """
    m = input_state.m
    ns = input_state.n
    ns = [ns] if isinstance(ns, int) else list(ns)

    for n in ns:
        if mask is not None:
            output_array = xq.FSArray(m, n, mask)
        else:
            output_array = xq.FSArray(m, n)
        for output_state in output_array:
            yield output_state


def max_photon_state_iterator(m: int, n_max: int):
    """
    Iterator on all possible output state on m modes with at most n_max photons

    :param m: number of modes
    :param n_max: maximum number of photons
    :return: list of BasicState
    """
    for n in range(n_max+1):
        output_array = xq.FSArray(m, n)
        for output_state in output_array:
            yield output_state


def tensorproduct(states: list[StateVector | BasicState]) -> StateVector | BasicState:
    r""" Computes states[0] * states[1] * ...
    """
    if len(states) == 1:
        return states[0]
    return tensorproduct(states[:-2] + [states[-2] * states[-1]])


class ProbabilityDistribution(defaultdict, ABC):
    """Time-Independent abstract probabilistic distribution of states
    """
    def __init__(self):
        super().__init__(float)

    def normalize(self):
        sum_probs = sum(list(self.values()))
        if sum_probs == 0:
            get_logger().warn("Unable to normalize a distribution with only null probabilities", channel.user)
            return
        for sv in self.keys():
            self[sv] /= sum_probs

    def add(self, obj, proba: float):
        if proba > global_params['min_p']:
            self[obj] += proba

    def __str__(self) -> str:
        return "{\n  "\
               + "\n  ".join(["%s: %s" % (str(k), v) for k, v in self.items()])\
               + "\n}"

    def __copy__(self) -> ProbabilityDistribution:
        distribution_copy = type(self)()
        for k, prob in self.items():
            distribution_copy[copy(k)] = prob
        return distribution_copy

    def __pow__(self, power) -> ProbabilityDistribution:
        return exponentiation_by_squaring(self, power)

    @abstractmethod
    def sample(self, count: int, non_null: bool = True) -> list:
        pass


class SVDistribution(ProbabilityDistribution):
    r"""
    Mixed state represented as a time-independent probabilistic distribution of StateVectors
    """
    def __init__(self, sv: BasicState | StateVector | dict | None = None):
        super().__init__()
        self._n_max = 0
        self._m = None
        if sv is not None:
            if isinstance(sv, (BasicState, StateVector)):
                self[sv] = 1
            elif isinstance(sv, dict):
                for k, v in sv.items():
                    self[k] = v
            else:
                raise TypeError(f"Unexpected type initializing SVDistribution {type(sv)}")

    def __setitem__(self, key, value):
        if isinstance(key, BasicState):
            key = StateVector(key)
        assert isinstance(key, StateVector), "SVDistribution key must be a BasicState or a StateVector"

        # number of modes verification
        if self._m is None:
            self._m = key.m
        if self._m != key.m:
            raise ValueError("Number of modes is not consistent")

        key.normalize()
        super().__setitem__(key, value)

        # Update max number of photons :
        n_max = max(key.n)
        if n_max > self._n_max:
            self._n_max = n_max

    def __getitem__(self, key):
        if isinstance(key, BasicState):
            key = StateVector(key)
        assert isinstance(key, StateVector), "SVDistribution key must be a BasicState or a StateVector"
        return super().__getitem__(key)

    def __mul__(self, other) -> SVDistribution:
        r"""Combines two `SVDistribution`

        :param other: State / distribution to multiply with
        :return: The result of the tensor product
        """
        if isinstance(other, SVDistribution):
            pass
        elif isinstance(other, (BasicState, StateVector)):
            other = SVDistribution(other)
        else:
            return NotImplemented
        if len(self) == 0:
            return other
        new_svd = SVDistribution()
        for sv1, proba1 in self.items():
            for sv2, proba2 in other.items():
                new_svd[sv1*sv2] = proba1 * proba2
        return new_svd

    def __rmul__(self, other) -> SVDistribution:
        if isinstance(other, SVDistribution):
            pass
        elif isinstance(other, (BasicState, StateVector)):
            other = SVDistribution(other)
        else:
            return NotImplemented
        return other * self

    def normalize(self):
        sum_probs = sum(list(self.values()))
        for sv in self.keys():
            self[sv] /= sum_probs

    def sample(self, count: int, non_null: bool = True) -> list[StateVector]:
        r""" Generate a sample StateVector from the `SVDistribution`

        :param non_null: excludes null states from the sample generation
        :param count: number of samples to draw
        :return: a list of :math:`count` samples
        """
        self.normalize()
        d = self
        if non_null:
            d = {sv: p for sv, p in self.items() if max(sv.n) != 0}
        if not d:
            raise RuntimeError("No state to sample from")
        states = list(d.keys())
        probs = list(d.values())
        results = random.choices(states, k=count, weights=probs)
        return list(results)

    @property
    def m(self) -> int:
        return self._m

    @property
    def n_max(self) -> int:
        return self._n_max

    @staticmethod
    def tensor_product(svd1, svd2, prob_threshold: float = 0) -> SVDistribution:
        """
        Compute the tensor product of two SVDistribution with an optional probability threshold
        """
        if len(svd1) == 0:
            return svd2
        new_dist = SVDistribution()
        for sv1, proba1 in svd1.items():
            for sv2, proba2 in svd2.items():
                if proba1 * proba2 < prob_threshold:
                    continue
                sv = sv1 * sv2
                new_dist[sv] += proba1 * proba2
        return new_dist

    @staticmethod
    def list_tensor_product(distributions: list[SVDistribution], prob_threshold: float = 0) -> SVDistribution:
        """Efficient tensor product for a list of distributions"""
        if len(distributions) == 0:
            return SVDistribution()

        if any(len(dist) == 0 for dist in distributions):
            get_logger().warn("Empty distribution in tensor product. Ignoring it", channel.user)
            distributions = [dist for dist in distributions if len(dist) > 0]

        while len(distributions) > 1:
            state_counts = [len(dist) for dist in distributions]

            # Find the index of lowest product with a sliding window and performs the tensor product with these.
            products = [state_counts[i] * state_counts[i + 1] for i in range(len(state_counts) - 1)]
            index = min(range(len(products)), key=products.__getitem__)

            dist = SVDistribution.tensor_product(distributions[index], distributions[index + 1], prob_threshold)
            distributions[index] = dist
            distributions.pop(index + 1)

        return distributions[0]


@dispatch(StateVector, annot_tag=str)
def anonymize_annotations(sv: StateVector, annot_tag: str = "a") -> StateVector:
    # This algo anonymizes annotations but is not enough to have superposed states exactly the same given the storage
    # order of BasicStates inside the StateVector
    m = sv.m
    annot_map = {}
    result = StateVector()
    for bs, pa in sv:
        s = [""] * m
        for i in range(bs.n):
            mode = bs.photon2mode(i)
            annot = bs.get_photon_annotation(i)
            if annot_map.get(str(annot)) is None:
                annot_map[str(annot)] = f"{{{annot_tag}:{len(annot_map)}}}"
            s[mode] += annot_map[str(annot)]
        result += StateVector("|" + ",".join([v and v or "0" for v in s]) + ">") * pa
    result.normalize()
    return result


@dispatch(SVDistribution, annot_tag=str)
def anonymize_annotations(svd: SVDistribution, annot_tag: str = "a") -> SVDistribution:
    sv_dist = defaultdict(lambda: 0)
    for k, p in svd.items():
        state = anonymize_annotations(k, annot_tag=annot_tag)
        sv_dist[state] += p
    return SVDistribution({k: v for k, v in sorted(sv_dist.items(), key=lambda x: -x[1])})


class BSDistribution(ProbabilityDistribution):
    r"""Time-Independent probabilistic distribution of Basic States
    """
    def __init__(self, d: BasicState | dict | None = None):
        super().__init__()
        self._m = None
        if d is not None:
            if isinstance(d, BasicState):
                self[d] = 1
            elif isinstance(d, dict):
                for k, v in d.items():
                    self[BasicState(k)] = v
            else:
                raise TypeError(f"Unexpected type initializing BSDistribution {type(d)}")

    def __setitem__(self, key, value):
        assert isinstance(key, BasicState), "BSDistribution key must be a BasicState"
        if self._m is None:
            self._m = key.m
        if self._m != key.m:
            raise ValueError("Number of modes is not consistent")
        super().__setitem__(key, value)

    def __getitem__(self, key) -> float:
        assert isinstance(key, BasicState), "BSDistribution key must be a BasicState"
        return super().__getitem__(key)

    def sample(self, count: int, non_null: bool = True) -> BSSamples:
        r""" Samples basic states from the `BSDistribution`

        :param count: number of samples to draw
        :param non_null: excludes null states from the sample generation
        :return: a list of :math:`count` samples
        """
        self.normalize()
        d = self
        if non_null:
            d = {bs: p for bs, p in self.items() if bs.n != 0}
        if not d:
            raise RuntimeError("No state to sample from")
        states = list(d.keys())
        probs = list(d.values())
        return random.choices(states, k=count, weights=probs)

    def __mul__(self, other) -> BSDistribution:
        if isinstance(other, BSDistribution):
            pass
        elif isinstance(other, BasicState):
            other = BSDistribution(other)
        else:
            return NotImplemented
        return BSDistribution.tensor_product(self, other)

    def __rmul__(self, other):
        if isinstance(other, BSDistribution):
            pass
        elif isinstance(other, BasicState):
            other = BSDistribution(other)
        else:
            return NotImplemented
        return BSDistribution.tensor_product(other, self)

    @staticmethod
    def tensor_product(bsd1: BSDistribution,
                       bsd2: BSDistribution,
                       merge_modes: bool = False,
                       prob_threshold: float = 0) -> BSDistribution:
        """
        Compute the tensor product of two BasicState Distribution
        """
        if len(bsd1) == 0:
            return bsd2
        new_dist = BSDistribution()
        for bs1, proba1 in bsd1.items():
            for bs2, proba2 in bsd2.items():
                if proba1 * proba2 < prob_threshold:
                    continue
                if merge_modes:
                    bs = bs1.merge(bs2)
                else:
                    bs = bs1 * bs2
                new_dist[bs] += proba1 * proba2
        return new_dist

    @staticmethod
    def list_tensor_product(distributions: list[BSDistribution],
                            merge_modes: bool = False,
                            prob_threshold: float = 0) -> BSDistribution:
        """
        Efficient tensor product for a list of BasicState Distribution.
         Can modify the distributions in place if merge_modes is False by adding empty modes.
         Performs `len(distributions) - 1` tensor products
         """
        if any(len(dist) == 0 for dist in distributions):
            get_logger().warn("Empty distribution in tensor product. Ignoring it", channel.user)
            distributions = [dist for dist in distributions if len(dist) > 0]

        if len(distributions) == 0:
            return BSDistribution()

        if not merge_modes:
            # Expanding the modes before allows performing the products in any order,
            # as the tensor product is commutative if merge_modes is True
            BSDistribution._expand_modes(distributions)

        while len(distributions) > 1:
            state_counts = [len(dist) for dist in distributions]  # strictly positive integers

            # Find the two smallest distributions and merge them
            idx_a = 0
            val_a = state_counts[idx_a]  # Smallest
            idx_b = 1
            val_b = state_counts[idx_b]  # Second smallest
            if val_b < val_a:
                val_a, val_b = val_b, val_a
                idx_a, idx_b = idx_b, idx_a
            for i, val in enumerate(state_counts[2:], 2):
                if val_a == 1 and val_b == 1:
                    break
                if val < val_a:
                    val_b, idx_b = val_a, idx_a
                    val_a = val
                    idx_a = i
                elif val < val_b:
                    val_b = val
                    idx_b = i

            dist = BSDistribution.tensor_product(distributions[idx_a], distributions[idx_b], True, prob_threshold)
            distributions[idx_a] = dist
            distributions.pop(idx_b)

        return distributions[0]

    @staticmethod
    def _expand_modes(distributions: list[BSDistribution]):
        """Add empty modes on the left and right of the distributions
         so their number of modes is the initial total number of modes"""
        current_m = 0
        total_m = sum(dist.m for dist in distributions)
        for i, dist in enumerate(distributions):
            m = dist.m
            dist = BSDistribution(BasicState(current_m * [0])) * dist
            dist *= BSDistribution(BasicState((total_m - dist.m) * [0]))
            distributions[i] = dist
            current_m += m

    @property
    def m(self) -> int:
        return self._m

    def photon_threshold_simplification(self, photon_threshold: int) -> BSDistribution:
        r""" Simplify this `BSDistribution` with a photon threshold for each mode
        (ex: |0,3,0,0> -> |0,1,0,0> if photon_threshold=1)
        These "coarse grain" simplification methods can be used to decrease the number of components of a given distribution.

        :param photon_threshold: the maximum number of photons per mode
        :return: the simplified distribution
        """
        simplified_distribution = BSDistribution()
        for bs, p in self.items():
            bs = BasicState([min(s, photon_threshold) for s in bs]) # for each mode, keep at most 'photon_threshold' photons
            simplified_distribution.add(bs, p)
        return simplified_distribution

    def group_modes_simplification(self, group_size: int) -> BSDistribution:
        r""" Simplify this `BSDistribution` by grouping modes
        (ex: |1,3,0,0> -> |4,0> if group_size=2)
        These "coarse grain" simplification methods can be used to decrease the number of components of a given distribution.

        :param group_size: the size of the groups of modes
        :return: the simplified distribution
        """
        simplified_distribution = BSDistribution()
        for bs, p in self.items():
            bs = BasicState([sum(bs[group_size*k:group_size*(k+1)]) for k in range(-(len(bs)//-group_size))]) # group modes by groups of size 'group_size'.
            # -(len(bs)//-group_size) is just the ceiling of len(bs)/group_size. The case group_size*(k+1) > len(bs) is correctly managed in python.
            simplified_distribution.add(bs, p)
        return simplified_distribution


def filter_distribution_photon_count(bsd: BSDistribution, min_photons_filter: int) -> tuple[BSDistribution, float]:
    """
    Filter the states of a BSDistribution to keep only those having state.n >= min_photons_filter

    :param bsd: the BSDistribution to filter out
    :param min_photons_filter: the minimum number of photons required to keep a state
    :return: a tuple containing the normalized filtered BSDistribution and the probability that the state is kept
    """
    if min_photons_filter == 0:
        return bsd, 1

    res = BSDistribution({state: prob for state, prob in bsd.items() if state.n >= min_photons_filter})
    perf = sum(res.values())

    if len(res):
        res.normalize()
    return res, perf
