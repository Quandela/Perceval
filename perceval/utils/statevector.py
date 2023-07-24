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
import itertools
from multipledispatch import dispatch
from typing import Dict, List, Union, Tuple, Optional
from deprecated import deprecated

from .format import simple_complex
from .globals import global_params
import numpy as np
import sympy as sp

from exqalibur import FockState, FSArray


def _fockstate_add(self, other):
    return StateVector(self) + other

def _fockstate_sub(self, other):
    return StateVector(self) - other

def _fockstate_pow(self, power: int):
    bs = self.__copy__()
    for i in range(power - 1):
        bs = bs * self
    return bs

def _fockstate_partition(self, distribution_photons: List[int]):
    r"""Given a distribution of photon, find all possible partition of the BasicState - disregard possible annotation

    :param distribution_photons:
    :return:
    """
    def _partition(one_list: list, distribution: list, current: list, all_res: list):
        if len(distribution) == 0:
            all_res.append(copy(current))
            return
        for one_subset in itertools.combinations(one_list, distribution[0]):
            current.append(one_subset)
            _partition(list(set(one_list)-set(one_subset)), distribution[1:], current, all_res)
            current.pop()

    all_photons = list(range(self.n))
    partitions_idx = []
    _partition(all_photons, distribution_photons, [], partitions_idx)
    partitions_states = set()
    for partition in partitions_idx:
        o_state = []
        for a_subset in partition:
            state = [0] * self.m
            for photon_id in a_subset:
                state[self.photon2mode(photon_id)] += 1
            o_state.append(BasicState(state))
        partitions_states.add(tuple(o_state))
    return list(partitions_states)


# Define BasicState as exqalibur FockState + redefine some methods
BasicState = FockState
BasicState.__add__ = _fockstate_add
BasicState.__sub__ = _fockstate_sub
BasicState.__pow__ = _fockstate_pow  # Using issue #210 fix before moving the fix to exqalibur
BasicState.partition = _fockstate_partition  # TODO use the cpp version of this call


def allstate_iterator(input_state: Union[BasicState, StateVector], mask=None) -> BasicState:
    """Iterator on all possible output states compatible with mask generating StateVector

    :param input_state: a given input state vector
    :param mask: an optional mask
    :return: list of output_state
    """
    m = input_state.m
    ns = input_state.n
    if not isinstance(ns, list):
        ns = [ns]
    for n in ns:
        if mask is not None:
            output_array = FSArray(m, n, mask)
        else:
            output_array = FSArray(m, n)
        for output_state in output_array:
            yield BasicState(output_state)


class AnnotatedBasicState(FockState):
    r"""Deprecated in version 0.7.0. Use BasicState instead.
    """

    @deprecated(version="0.7", reason="use BasicState instead")
    def __init__(self, *args, **kwargs):
        super(AnnotatedBasicState, self).__init__(*args, **kwargs)


class StateVector(defaultdict):
    """
    A StateVector is a (complex) linear combination of Basic States
    """

    def __init__(self,
                 bs: Union[BasicState, List[int], str, None] = None,
                 photon_annotations: Dict[int, List] = None):
        r"""Init of a StateVector from a BasicState, or from BasicState constructor
        :param bs: a BasicState, or `BasicState` constructor,
                    None used for internal purpose
        :param photon_annotations: photon annotation dictionary
        """
        super(StateVector, self).__init__(float)
        self.m = None
        if bs is not None:
            if not isinstance(bs, BasicState):
                bs = BasicState(bs, photon_annotations or {})
            else:
                assert photon_annotations is None, "cannot add photon annotations to BasicState"
            self[bs] = 1
        self._normalized = True
        self._has_symbolic = False

    def __eq__(self, other):
        if not isinstance(other, StateVector):
            return False
        self.normalize()
        other.normalize()
        return super().__eq__(other)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __rmul__(self, other):
        r"""Multiply a StateVector by a numeric value, right side
        """
        return self*other

    def __getitem__(self, key):
        if isinstance(key, int):
            return list(self.keys())[key]
        assert isinstance(key, BasicState), "StateVector keys should be Basic States"
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        assert isinstance(key, BasicState), "StateVector keys should be Basic States"
        self._normalized = False
        if self.m is None:
            self.m = key.m
        return super().__setitem__(key, value)

    def __len__(self):
        self.normalize()
        return super().__len__()

    def __iter__(self):
        self.normalize()
        return super().__iter__()

    def __mul__(self, other):
        r"""Multiply a StateVector by a numeric value prior in a linear combination,
        or computes the tensor product between two StateVectors
        """
        if isinstance(other, (StateVector, BasicState)):
            if isinstance(other, BasicState):
                other = StateVector(other)
            sv = StateVector()
            if not other:
                return self
            if not self:
                return other
            sv.update({l_state * r_state: self[l_state] * other[r_state] for r_state in other for l_state in self})
            sv.m = self.m + other.m
            sv._has_symbolic = (other._has_symbolic or self._has_symbolic)
            sv._normalized = (other._normalized and self._normalized)
            return sv

        assert isinstance(other, (int, float, complex, np.number, sp.Expr)), "normalization factor has to be numeric"
        copy_state = StateVector(None)
        # multiplying - the outcome is a non-normalized StateVector
        for state, amplitude in self.items():
            copy_state[state] = amplitude*other
        if other != 1:
            copy_state._normalized = False
        copy_state.m = self.m
        if isinstance(other, sp.Expr):
            self._has_symbolic = True
        return copy_state

    def __pow__(self, power):
        # Fast exponentiation
        binary = [int(i) for i in bin(power)[2:]]
        binary.reverse()
        power_state = self
        out = StateVector()
        for i in range(len(binary)):
            if binary[i] == 1:
                out *= power_state
            if i != len(binary) - 1:
                power_state *= power_state
        return out

    def __copy__(self):
        sv_copy = StateVector(None)
        for k, v in self.items():
            sv_copy[copy(k)] = v
        sv_copy._has_symbolic = self._has_symbolic
        sv_copy._normalized = self._normalized
        sv_copy.m = self.m
        return sv_copy

    def __add__(self, other):
        r"""Add two StateVectors"""
        assert isinstance(other, (StateVector, BasicState)), "addition requires states"
        if self.m is None:
            self.m = other.m
        assert other.m == self.m, "invalid mix of different modes"
        copy_state = copy(self)
        if not isinstance(other, StateVector):
            other = StateVector(other)
        # multiplying - the outcome is a non-normalized StateVector
        for state, amplitude in other.items():
            if state not in copy_state:
                copy_state[state] = amplitude
            else:
                copy_state[state] += amplitude
        copy_state._normalized = False
        copy_state.m = self.m
        if other._has_symbolic:
            copy_state._has_symbolic = True
        return copy_state

    def __sub__(self, other):
        r"""Sub two StateVectors"""
        if isinstance(other, BasicState):
            other = StateVector(other)
        return self + -1 * other

    def sample(self) -> BasicState:
        r"""Sample a single BasicState from the statevector.
        It does not perform a measure - so do not change the value of the statevector

        :return: a BasicState
        """
        p = random.random()
        idx = 0
        keys = list(self.keys())
        self.normalize()
        while idx < len(keys)-1:
            p = p - abs(self[keys[idx]])**2
            if p < 0:
                break
            idx += 1
        return BasicState(keys[idx])

    def samples(self, shots: int) -> List[BasicState]:
        r"""Generate a list of samples.
        It does not perform a measure - so do not change the value of statevector.
        This function is more efficient than run :math:$shots$ times :method:sample

        :param shots: the number of samples
        :return: a list of BasicState
        """
        self.normalize()
        weight = [abs(self[key])**2 for key in self.keys()]
        rng = np.random.default_rng()
        return [BasicState(x) for x in rng.choice(list(self.keys()), shots, p=weight)]

    def measure(self, modes: Union[int, List[int]]) -> Dict[BasicState, Tuple[float, StateVector]]:
        r"""perform a measure on one or multiple modes and collapse the remaining statevector. The resulting
        statevector are not normalised by default.

        :param modes: the mode to measure
        :return: a dictionary - key is the possible measures, values are pairs (probability, BasicState vector)
        """
        self.normalize()
        if isinstance(modes, int):
            modes = [modes]
        map_measure_sv = defaultdict(lambda: [0, StateVector()])
        for s, pa in self.items():
            out = []
            remaining_state = []
            p = abs(pa)**2
            for i in range(self.m):
                if i in modes:
                    out.append(s[i])
                else:
                    remaining_state.append(s[i])
            map_measure_sv[BasicState(out)][0] += p
            map_measure_sv[BasicState(out)][1] += pa*StateVector(remaining_state)
        return {k: tuple(v) for k, v in map_measure_sv.items()}

    @property
    def n(self):
        r"""list the possible values of n in the different states"""
        return list(set([st.n for st in self.keys()]))

    def normalize(self):
        r"""Normalize a state vector"""
        if not self._normalized:
            norm = 0
            to_remove = []
            for key in self.keys():
                if (isinstance(self[key], (complex, float, int))
                        and abs(self[key]) < global_params["min_complex_component"]) or self[key] == 0:
                    to_remove.append(key)
                else:
                    norm += abs(self[key])**2
            for key in to_remove:
                del self[key]
            norm = norm**0.5
            nkey = len(self.keys())
            for key in self.keys():
                if nkey == 1:
                    self[key] = 1
                else:
                    self[key] /= norm
            self._normalized = True

    def __str__(self, nsimplify=True):
        if not self.keys():
            return "|>"
        self_copy = copy(self)
        self_copy.normalize()
        ls = []
        for key, value in self_copy.items():
            if value == 1:
                ls.append(str(key))
            else:
                if isinstance(value, sp.Expr):
                    ls.append(str(value) + "*" + str(key))
                else:
                    if nsimplify:
                        value = simple_complex(value)[1]
                        if value[1:].find("-") != -1 or value.find("+") != -1:
                            value = f"({value})"
                    else:
                        value = str(value)
                    ls.append( value + "*" + str(key))
        return "+".join(ls).replace("+-", "-")

    def __hash__(self):
        return self.__str__(nsimplify=False).__hash__()


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
        # Fast exponentiation
        binary = [int(i) for i in bin(power)[2:]]
        binary.reverse()
        power_distrib = self
        out = type(self)()
        for i in range(len(binary)):
            if binary[i] == 1:
                out *= power_distrib
            if i != len(binary) - 1:
                power_distrib *= power_distrib
        return out

    @abstractmethod
    def sample(self, count: int, non_null: bool = True):
        pass


class SVDistribution(ProbabilityDistribution):
    r"""Time-Independent Probabilistic distribution of StateVectors
    """
    def __init__(self, sv: Optional[BasicState, StateVector, Dict] = None):
        super().__init__()
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
        key.normalize()
        super().__setitem__(key, value)

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
        states = list(d.keys())
        probs = list(d.values())
        rng = np.random.default_rng()
        results = rng.choice(states, count, p=np.array(probs) / sum(probs))
        return list(results)


@dispatch(StateVector, annot_tag=str)
def anonymize_annotations(sv: StateVector, annot_tag: str = "a"):
    m = sv.m
    annot_map = {}
    result = StateVector()
    for bs, pa in sv.items():
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

    def __init__(self, d: Optional[BasicState, Dict] = None):
        super().__init__()
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
        states = list(d.keys())
        probs = list(d.values())
        rng = np.random.default_rng()
        results = rng.choice(states, count, p=probs)
        # numpy transforms iterables of ints to a nparray in rng.choice call
        # Thus, we need to convert back the results to BasicStates
        output = BSSamples()
        for s in results:
            output.append(BasicState(s))
        return output

    def __mul__(self, other):
        return BSDistribution.tensor_product(self, other)

    @staticmethod
    def tensor_product(bsd1, bsd2, merge_modes: bool = False, prob_threshold: float = 0):
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


class BSCount(defaultdict):
    def __init__(self, d: Optional[Dict] = None):
        super().__init__(int)
        if d is not None:
            for k, v in d.items():
                self[BasicState(k)] = v

    def __setitem__(self, key, value):
        assert isinstance(key, BasicState), "BSCount key must be a BasicState"
        assert isinstance(value, int) and value >= 0, "Count must be a positive integer"
        super().__setitem__(key, value)

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
