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

from __future__ import annotations

import random
import warnings
from collections import defaultdict
from copy import copy
import itertools
import re
from typing import Dict, List, Union, Tuple, Optional
from deprecated import deprecated

from .matrix import Matrix
from .format import simple_complex, simple_float
from .globals import global_params
from .polarization import Polarization
import numpy as np
import sympy as sp

from quandelibc import FockState, Annotation, FSArray


class BasicState(FockState):
    r"""Basic states
    """

    def __init__(self, *args, **kwargs):
        super(BasicState, self).__init__(*args, **kwargs)

    def __len__(self):
        return self.m

    def __copy__(self):
        return BasicState(self)

    def __add__(self, o):
        return StateVector(self) + o

    def __sub__(self, o):
        if not isinstance(o, StateVector):
            o = StateVector(o)
        return StateVector(self) - o

    def separate_state(self):
        return [BasicState(s) for s in super(BasicState, self).separate_state()]

    def __mul__(self, s):
        if isinstance(s, StateVector):
            return StateVector(self) * s
        return BasicState(super(BasicState, self).__mul__(s))

    def __pow__(self, power):
        return BasicState(power * list(self))

    def __getitem__(self, item):
        it = super().__getitem__(item)
        if isinstance(it, FockState):
            it = BasicState(it)
        return it

    def set_slice(self, slice, state):
        return BasicState(super().set_slice(slice, state))

    def partition(self, distribution_photons: List[int]):
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
        for output_idx, output_state in enumerate(output_array):
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
                 photon_annotations: Dict[int, str] = None):
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

    def __rmul__(self, other):
        r"""Multiply a StateVector by a numeric value, right side
        """
        return self*other

    def __getitem__(self, key):
        if isinstance(key, int):
            return list(self.keys())[key]
        assert isinstance(key, BasicState), "SVState keys should be Basic States"
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        assert isinstance(key, BasicState), "SVState keys should be Basic States"
        self._normalized = False
        if self.m is None:
            self.m = key.m
        return super().__setitem__(key, value)

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
            sv_copy[k] = v
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
        r"""perform a measure on one or multiple modes and collapse the remaining statevector

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
        r"""Normalize a non-normalized BasicState"""
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
            for key in self.keys():
                if len(self) == 1:
                    self[key] = 1
                else:
                    self[key] /= norm
            self._normalized = True

    def __str__(self):
        if not self:
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
                    value = simple_complex(value)[1]
                    if value[1:].find("-") != -1 or value.find("+") != -1:
                        value = "("+value+")"
                    ls.append( value + "*" + str(key))
        return "+".join(ls).replace("+-", "-")

    def __hash__(self):
        return self.__str__().__hash__()


def tensorproduct(states: List[Union[StateVector, BasicState]]):
    r""" Computes states[0] * states[1] * ...
    """
    if len(states) == 1:
        return states[0]
    return tensorproduct(states[:-2] + [states[-2] * states[-1]])


class ProbabilityDistribution(defaultdict):
    """Time-Independent abstract probabilistic distribution of StateVectors
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
        self[obj] += proba

    def __str__(self):
        return "{\n  "\
               + "\n  ".join(["%s: %s" % (str(k), simple_float(v, nsimplify=True)[1]) for k, v in self.items()])\
               + "\n}"

    def __copy__(self):
        distribution_copy = type(self)()
        for k, prob in self.items():
            distribution_copy[copy(k)] = prob
        return distribution_copy


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

    def __mul__(self, svd):
        r"""Combines two `SVDistribution`

        :param svd:
        :return:
        """
        if len(self) == 0:
            return svd
        new_svd = SVDistribution()
        for sv1, proba1 in self.items():
            for sv2, proba2 in svd.items():
                # assert len(sv1) == 1 and len(sv2) == 1, "can only combine basic states"
                new_svd[sv1*sv2] = proba1 * proba2

        return new_svd

    def __pow__(self, power):
        # Fast exponentiation
        binary = [int(i) for i in bin(power)[2:]]
        binary.reverse()
        power_svd = self
        out = SVDistribution()
        for i in range(len(binary)):
            if binary[i] == 1:
                out *= power_svd
            if i != len(binary) - 1:
                power_svd *= power_svd
        return out

    def normalize(self):
        sum_probs = sum(list(self.values()))
        for sv in self.keys():
            self[sv] /= sum_probs

    def sample(self, count: int = 1, non_null: bool = True) -> List[StateVector]:
        r""" Generate a sample StateVector from the `SVDistribution`

        :param non_null: excludes null states from the sample generation
        :param count: number of samples to draw
        :return: if :math:`count=1` a single sample, if :math:`count>1` a list of :math:`count` samples
        """
        self.normalize()
        d = self
        if non_null:
            d = {sv: p for sv, p in self.items() if max(sv.n) != 0}
        states = list(d.keys())
        probs = list(d.values())
        rng = np.random.default_rng()
        results = rng.choice(states, count, p=np.array(probs) / sum(probs))
        if len(results) == 1:
            return results[0]
        return list(results)


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

    def samples(self, count: int = 1) -> BSSamples:
        r""" Samples basic states from the `BSDistribution`

        :param count: number of samples to draw
        :return: if :math:`count=1` a single sample, if :math:`count>1` a list of :math:`count` samples
        """
        self.normalize()
        states = list(self.keys())
        probs = list(self.values())
        rng = np.random.default_rng()
        results = rng.choice(states, count, p=probs)
        # numpy transforms iterables of ints to a nparray in rng.choice call
        # Thus, we need to convert back the results to BasicStates
        output = BSSamples()
        for s in results:
            output.append(BasicState(s))
        return output


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


def _rec_build_spatial_output_states(lfs: list, output: list):
    if len(lfs) == 0:
        yield BasicState(output)
    else:
        if lfs[0] == 0:
            yield from _rec_build_spatial_output_states(lfs[1:], output+[0, 0])
        else:
            for k in range(lfs[0]+1):
                yield from _rec_build_spatial_output_states(lfs[1:], output+[k, lfs[0]-k])


def build_spatial_output_states(state: BasicState):
    yield from _rec_build_spatial_output_states(list(state), [])


def _is_orthogonal(v1, v2, use_symbolic):
    if use_symbolic:
        orth = sp.conjugate(v1[0]) * v2[0] + sp.conjugate(v1[1]) * v2[1]
        return orth == 0
    orth = np.conjugate(v1[0]) * v2[0] + np.conjugate(v1[1]) * v2[1]
    return abs(orth) < 1e-6


def convert_polarized_state(state: BasicState,
                            use_symbolic: bool = False,
                            inverse: bool = False) -> Tuple[BasicState, Matrix]:
    r"""Convert a polarized BasicState into an expanded BasicState vector

    :param inverse:
    :param use_symbolic:
    :param state:
    :return:
    """
    idx = 0
    input_state = []
    prep_matrix = None
    for k_m in range(state.m):
        input_state += [0, 0]
        if state[k_m]:
            vectors = []
            for k_n in range(state[k_m]):
                # for each state we can handle up to two orthogonal vectors
                annot = state.get_photon_annotation(idx)
                idx += 1
                v_hv = Polarization(annot.get("P", complex(Polarization(0)))).project_eh_ev(use_symbolic)
                v_idx = None
                for i, v in enumerate(vectors):
                    if v == v_hv:
                        v_idx = i
                        break
                if v_idx is None:
                    if len(vectors) == 2:
                        raise ValueError("use statevectors to handle more than 2 orthogonal vectors")
                    if len(vectors) == 0 or _is_orthogonal(vectors[0], v_hv, use_symbolic):
                        v_idx = len(vectors)
                        vectors.append(v_hv)
                    else:
                        raise ValueError("use statevectors to handle non orthogonal vectors")
                input_state[-2+v_idx] += 1
            if vectors:
                eh1, ev1 = vectors[0]
                if len(vectors) == 1:
                    if use_symbolic:
                        eh2 = -sp.conjugate(ev1)
                        ev2 = sp.conjugate(eh1)
                    else:
                        eh2 = -np.conjugate(ev1)
                        ev2 = np.conjugate(eh1)
                else:
                    eh2, ev2 = vectors[1]
                if prep_matrix is None:
                    prep_matrix = Matrix.eye(2*state.m, use_symbolic)
                prep_state_matrix = Matrix([[eh1, eh2],
                                            [ev1, ev2]], use_symbolic)
                if inverse:
                    prep_state_matrix = prep_state_matrix.inv()
                prep_matrix[2*k_m:2*k_m+2, 2*k_m:2*k_m+2] = prep_state_matrix
    return BasicState(input_state), prep_matrix
