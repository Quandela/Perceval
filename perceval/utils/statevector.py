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
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from copy import copy
from multipledispatch import dispatch
from typing import Dict, List, Union, Optional

from .globals import global_params
from .qmath import exponentiation_by_squaring
import exqalibur as xq

BasicState = xq.FockState
StateVector = xq.StateVector


def allstate_iterator(input_state: Union[BasicState, StateVector], mask=None) -> BasicState:
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
        output_array = xq.FSArray(m,n)
        for output_state in output_array:
            yield output_state


def tensorproduct(states: List[Union[StateVector, BasicState]]):
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
            warnings.warn("Unable to normalize a distribution with only null probabilities")
            return
        for sv in self.keys():
            self[sv] /= sum_probs

    def add(self, obj, proba: float):
        if proba > global_params['min_p']:
            self[obj] += proba

    def __str__(self):
        return "{\n  "\
               + "\n  ".join(["%s: %s" % (str(k), v) for k, v in self.items()])\
               + "\n}"

    def __copy__(self):
        distribution_copy = type(self)()
        for k, prob in self.items():
            distribution_copy[copy(k)] = prob
        return distribution_copy

    def __pow__(self, power):
        return exponentiation_by_squaring(self, power)

    @abstractmethod
    def sample(self, count: int, non_null: bool = True):
        pass


class SVDistribution(ProbabilityDistribution):
    r"""Time-Independent Probabilistic distribution of StateVectors
    """
    def __init__(self, sv: Optional[BasicState, StateVector, Dict] = None):
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

    def __mul__(self, other):
        r"""Combines two `SVDistribution`

        :param other: State / distribution to multiply with
        :return: The result of the tensor product
        """
        if isinstance(other, (BasicState, StateVector)):
            other = SVDistribution(other)
        if len(self) == 0:
            return other
        new_svd = SVDistribution()
        for sv1, proba1 in self.items():
            for sv2, proba2 in other.items():
                new_svd[sv1*sv2] = proba1 * proba2
        return new_svd

    def normalize(self):
        sum_probs = sum(list(self.values()))
        for sv in self.keys():
            self[sv] /= sum_probs

    def sample(self, count: int, non_null: bool = True) -> List[StateVector]:
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
    def m(self):
        return self._m

    @property
    def n_max(self):
        return self._n_max

    @staticmethod
    def tensor_product(svd1, svd2, prob_threshold: float = 0):
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


@dispatch(StateVector, annot_tag=str)
def anonymize_annotations(sv: StateVector, annot_tag: str = "a"):
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
def anonymize_annotations(svd: SVDistribution, annot_tag: str = "a"):
    sv_dist = defaultdict(lambda: 0)
    for k, p in svd.items():
        state = anonymize_annotations(k, annot_tag=annot_tag)
        sv_dist[state] += p
    return SVDistribution({k: v for k, v in sorted(sv_dist.items(), key=lambda x: -x[1])})


class BSDistribution(ProbabilityDistribution):
    r"""Time-Independant probabilistic distribution of Basic States
    """
    def __init__(self, d: Optional[BasicState, Dict] = None):
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

    def __getitem__(self, key):
        assert isinstance(key, BasicState), "BSDistribution key must be a BasicState"
        return super().__getitem__(key)

    def sample(self, count: int, non_null: bool = True) -> BSSamples:
        r""" Samples basic states from the `BSDistribution`

        :param count: number of samples to draw
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

    def __mul__(self, other):
        return BSDistribution.tensor_product(self, other)

    @staticmethod
    def tensor_product(bsd1, bsd2, merge_modes: bool = False, prob_threshold: float = 0):
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

    @property
    def m(self):
        return self._m



class BSCount(defaultdict):
    r"""Container that counts basic state events
    """
    def __init__(self, d: Optional[Dict] = None):
        super().__init__(int)
        if d is not None:
            for k, v in d.items():
                self[BasicState(k)] = v

    def __setitem__(self, key, value):
        assert isinstance(key, BasicState), "BSCount key must be a BasicState"
        assert value >= 0, "Count must be a positive value"
        super().__setitem__(key, int(value))

    def __getitem__(self, key):
        assert isinstance(key, BasicState), "BSCount key must be a BasicState"
        return super().__getitem__(key)

    def add(self, obj, count: int):
        if count != 0:
            self[obj] += count

    def total(self):
        return sum(list(self.values()))

    def __str__(self):
        return "{\n  " + "\n  ".join([f"{k}: {v}" for k, v in self.items()]) + "\n}"


class BSSamples(list):
    r"""Container that stores samples in a time ordered way
    """
    def __setitem__(self, index, item):
        assert isinstance(item, BasicState), "BSSamples key must be a BasicState"
        super().__setitem__(index, item)

    def __str__(self):
        sz = len(self)
        n_to_display = min(sz, 10)
        s = '[' + ', '.join([str(bs) for bs in self[:n_to_display]])
        if sz > n_to_display:
            s += f', ... (size={sz})'
        s += ']'
        return s
