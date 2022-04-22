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
from collections import defaultdict
from copy import copy
import itertools
import re
from typing import Dict, List, Union, Tuple, Optional

from tabulate import tabulate

from perceval.utils import simple_complex, simple_float, Matrix, global_params
from .polarization import Polarization
import numpy as np
import sympy as sp

from quandelibc import FockState


class BasicState(FockState):
    r"""Basic states
    """
    def __init__(self, *args, **kwargs):
        super(BasicState, self).__init__(*args, **kwargs)

    @property
    def has_polarization(self):
        return False

    def __add__(self, o):
        return StateVector(self) + o

    def __sub__(self, o):
        if not isinstance(o, StateVector):
            o = StateVector(o)
        return StateVector(self) - o


class Annotations(dict):
    r"""Photon annotations
    """

    def __init__(self, annots: Union[Dict, Annotations]):
        super().__init__()
        for k, v in annots.items():
            self[k] = copy(v)

    @staticmethod
    def parse_annotation(s):
        l_s = s.split(",")
        annots = {}
        for lk in l_s:
            split_lk = lk.split(":")
            if len(split_lk) != 2:
                raise ValueError("invalid annotation format: %s" % s)
            k = split_lk[0]
            v = split_lk[1].replace("،", ",")
            if k == "P":
                v = Polarization.parse(v)
            else:
                try:
                    v = int(split_lk[1])
                except ValueError:
                    try:
                        v = float(split_lk[1])
                    except ValueError:
                        try:
                            v = sp.S(split_lk[1])
                            if v.free_symbols:
                                v = split_lk[1]
                        except ValueError:
                            pass
            if split_lk[0] in annots:
                raise ValueError("same annotation is declared twice `%s` in %s", split_lk[0], s)
            annots[split_lk[0]] = v
        return annots

    def check_special_annotations(self):
        for k, v in self.items():
            if k == "P":
                if isinstance(v, str):
                    self[k] = float(v)

    def __str__(self):
        """compact string representation"""
        represented = []
        for k, v in self.items():
            if isinstance(v, tuple):
                v = str(v).replace(" ", "")
            represented.append("{0}:{1}".format(k, v))
        return "{" + ",".join(represented) + "}"


class AnnotatedBasicState(BasicState):
    r"""Extends `BasicState` with annotations"""

    def __init__(self,
                 bs: Union[BasicState, List[int], str],
                 photon_annotations: Dict[int, Union[Dict, Annotations]] = None):
        r"""Definition of a State element

        :param bs: a `BasicState` initializer: string, list, or a Basic State
        :param photon_annotations: for each photon in the state, indexed by left-to-right position in `BasicState`
            provides an annotation dictionary. Photon index is 1-based.
        """
        if isinstance(bs, str) and bs.find("{") != -1:
            assert not photon_annotations, "cannot use photon annotations together with serialized annotations %s" % bs
            # parse annotated states
            if bs[0] != "|" or bs[-1] != ">":
                raise ValueError("incorrect syntax for annotated state: %s" % bs)
            v = bs[1:-1]
            # protect all commas inside annotations, pair separator is internally '،'
            v = re.sub(r"(\([^)]*),(.*?\))", r"\1،\2", v)
            photon_index = 1
            mode_index = 0
            states = []
            photon_annotations = {}
            # in general (\d?{})*\d,...
            while v:
                if v[0] == ",":
                    raise ValueError("mode cannot be empty: %s" % bs)
                m = re.match(r"(\d+)(.*)", v)
                if m:
                    counter = int(m.group(1))
                    v = m.group(2)
                else:
                    if not len(v) or v[0] != "{":
                        raise ValueError("invalid annotation: " % bs)
                    counter = 1
                if v and v[0] == "{":
                    if counter == 0:
                        raise ValueError("annotations can not be on 0 photons: %s" % bs)
                    m = re.match(r"{(.*?)}(.*)", v)
                    if m is None:
                        raise ValueError("non-closed annotation: %s" % bs)
                    annotation = Annotations.parse_annotation(m.group(1))
                    v = m.group(2)
                else:
                    annotation = {}
                for _ in range(counter):
                    if annotation:
                        photon_annotations[photon_index] = annotation
                    photon_index += 1
                if len(states) <= mode_index:
                    states.append(0)
                states[mode_index] += counter
                if v and v[0] == ",":
                    mode_index += 1
                    v = v[1:]
                    if not v:
                        raise ValueError("mode cannot be empty: %s" % bs)
            bs = states

        super().__init__(bs)
        self._annotations = None
        if photon_annotations is not None:
            self._annotations = []
            for _ in range(self.n):
                self._annotations.append(Annotations({}))
            for k, v in photon_annotations.items():
                self._annotations[k-1] = Annotations(v)

    @property
    def has_annotations(self) -> bool:
        r"""True is the BasicState uses annotations
        """
        return self._annotations is not None

    @property
    def has_polarization(self) -> bool:
        if self._annotations is None:
            return False
        for annot in self._annotations:
            if "P" in annot:
                return True
        return False

    def clear(self) -> AnnotatedBasicState:
        r"""Clear all annotations on current Basic States

        :return: returns the current cleared object
        """
        self._annotations = None
        return self

    def get_mode_annotations(self, k: int) -> Tuple[Annotations]:
        r"""Retrieve annotations of the photons in the given mode

        :param k: the mode
        :return: tuple annotation list
        """
        annots = []
        if self[k] != 0:
            photon_idx = self.mode2photon(k)
            while len(annots) < self[k]:
                annots.append(self._annotations is not None and self._annotations[photon_idx] or Annotations({}))
                photon_idx += 1
        return tuple(annots)

    def get_photon_annotations(self, pk: int) -> Annotations:
        r"""Retrieve annotations of the k-th photon

        :param pk: the photon 1-based index
        :return: tuple annotation list
        """
        if self._annotations is None:
            return Annotations({})
        return self._annotations[pk-1]

    def set_photon_annotations(self, pk: int, annots: Union[Dict, Annotations]) -> None:
        r"""Set annotations of the k-th photon (1-based index)

        :param pk: the photon 1-based index
        :param annots: the annotations
        """
        if self._annotations is None:
            self._annotations = []
            for _ in range(self.n):
                self._annotations.append(Annotations({}))
        else:
            for k, v in annots.items():
                self._annotations[pk-1][k] = copy(v)

    def separate_state(self) -> List[AnnotatedBasicState]:
        r"""Separate an `AnnotatedBasicState` on states with indistinguishable photons

        :return: list of `AnnotatedBasicState` - might be the current state.
        """

        if self.n == 0:
            return [BasicState([0]*self.m)]

        def _annot_compatible(annot_ref, annot_add):
            new_annot = copy(annot_ref)
            for a_k, a_v in annot_add.items():
                if a_k == "P":
                    continue
                if a_k not in annot_ref:
                    new_annot[a_k] = a_v
                else:
                    if annot_ref[a_k] != a_v:
                        return False
            return new_annot

        # check which photon needs to be distinguished - if we have n photons, we might end-up with n partitions
        photon_groups = []
        for k in range(self.n):
            annot_k = self.get_photon_annotations(k+1)
            merged_photon = False
            for annot_photon in photon_groups:
                merge_annot = _annot_compatible(annot_photon[0], annot_k)
                if merge_annot is not False:
                    annot_photon[0] = merge_annot
                    annot_photon[1].append(k)
                    merged_photon = True
                    break
            if not merged_photon:
                photon_groups.append([annot_k, [k]])

        if len(photon_groups) == 1:
            return [self]
        # now iterate through photon_groups and generate corresponding states
        states = []
        for annot_photon in photon_groups:
            state = [0] * self.m
            for photon_id in annot_photon[1]:
                state[self.photon2mode(photon_id)] += 1
            states.append(BasicState(state))

        return states

    def partition(self, distribution_photons: List[int]):
        r"""Given a distribution of photon, find all possible partition of the state

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

    def __str__(self):
        """Textual Representation of a BasicState

        :return: string representation of BasicState
        """
        if self._annotations:
            ls = []
            for k in range(self.m):
                m = self.get_mode_annotations(k)
                if len(m) == 0:
                    ls.append("0")
                else:
                    ms = []
                    photons_annots = sorted([str(annot) for annot in m])
                    k = 0
                    while k < len(photons_annots):
                        m = k + 1
                        while m < len(photons_annots) and photons_annots[k] == photons_annots[m]:
                            m += 1
                        n = m-k
                        if photons_annots[k] == "{}":
                            ms.append(str(n))
                        else:
                            s = photons_annots[k]
                            if n > 1:
                                s = str(n)+s
                            ms.append(s)
                        k = m
                    ls.append("".join(ms))
            s = "|"+",".join(ls)+">"
        else:
            s = super(AnnotatedBasicState, self).__str__()
        return s

    def __mul__(self, a_bs):
        new_a_bs = AnnotatedBasicState(super(AnnotatedBasicState, self).__mul__(a_bs))
        if self.has_annotations or a_bs.has_annotations:
            new_a_bs._annotations = []
            for k in range(self.n):
                if self.has_annotations:
                    new_a_bs._annotations.append(self.get_photon_annotations(k+1))
                else:
                    new_a_bs._annotations.append(Annotations({}))
            for k in range(a_bs.n):
                if a_bs.has_annotations:
                    new_a_bs._annotations.append(a_bs.get_photon_annotations(k+1))
                else:
                    new_a_bs._annotations.append(Annotations({}))
        return new_a_bs

    def __hash__(self):
        if self.has_annotations:
            return self.__str__().__hash__()
        else:
            return super(AnnotatedBasicState, self).__hash__()


class StateVector(defaultdict):
    """
    A StateVector is a (complex) linear combination of annotated Basic States
    """

    def __init__(self,
                 bs: Union[BasicState, List[int], str, AnnotatedBasicState, None] = None,
                 photon_annotations: Dict[int, Union[Dict, Annotations]] = None):
        r"""Init of a StateVector from a BasicState, or from BasicState constructor
        :param bs: an annotated basic state, a basic state, or a `BasicState` constructors,
                    None used for internal purpose
        :param photon_annotations: photon annotation dictionary
        """
        super(StateVector, self).__init__(float)
        self.m = None
        if bs is not None:
            if not isinstance(bs, AnnotatedBasicState):
                bs = AnnotatedBasicState(bs, photon_annotations)
            else:
                assert photon_annotations is None, "cannot add photon annotations to AnnotatedBasicState"
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
        assert isinstance(key, BasicState), "SVState keys should be (annotated) basic states"
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        assert isinstance(key, BasicState), "SVState keys should be (annotated) basic states"
        self._normalized = False
        if self.m is None:
            self.m = key.m
        return super().__setitem__(key, value)

    def __iter__(self):
        self._normalize()
        return super().__iter__()

    def __mul__(self, other):
        r"""Multiply a StateVector by a numeric value prior in a linear combination
        """
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
        assert isinstance(other, (StateVector, AnnotatedBasicState, BasicState)), "addition requires states"
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

    @property
    def n(self):
        r"""list the possible values of n in the different states"""
        return list(set([st.n for st in self.keys()]))

    def _normalize(self):
        r"""Normalize a non-normalized state"""
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
        self._normalize()
        ls = []
        for key, value in self.items():
            if value == 1:
                ls.append(str(key))
            else:
                if isinstance(value, sp.Expr):
                    ls.append(str(value) + "*" + str(key))
                else:
                    ls.append(simple_complex(value)[1] + "*" + str(key))
        return "+".join(ls).replace("+-", "-")

    def __hash__(self):
        return self.__str__().__hash__()


class SVTimeSequence:
    r"""Sequence in time of state-vector
    """

    def __init__(self):
        pass


class SVDistribution(defaultdict):
    r"""Time-Independent Probabilistic distribution of StateVectors
    """
    def __init__(self, sv: Optional[StateVector] = None):
        super(SVDistribution, self).__init__(float)
        if sv is not None:
            self[sv] = 1

    def add(self, sv: StateVector, proba: float):
        self[sv] += proba

    def __setitem__(self, key, value):
        assert isinstance(key, StateVector), "SVDistribution keys are state vectors"
        super().__setitem__(key, value)

    def __getitem__(self, key):
        assert isinstance(key, StateVector), "SVDistribution keys are state vectors"
        return super().__getitem__(key)

    def __mul__(self, svd):
        r"""Combines two `SVDistribution`

        :param svd:
        :return:
        """
        new_svd = SVDistribution()
        for sv1, proba1 in self.items():
            for sv2, proba2 in svd.items():
                assert len(sv1) == 1 and len(sv2) == 1, "can only combine basic states"
                new_svd[StateVector(sv1[0]*sv2[0])] = proba1 * proba2

        return new_svd

    def sample(self, k: int = 1, non_null: bool = True) -> List[StateVector]:
        r""" Generate a sample StateVector from the `SVDistribution`

        :param non_null: excludes null states from the sample generation
        :param k: number of samples to draw
        :return: if :math:`k=1` a single sample, if :math:`k>1` a list of :math:`k` samples
        """
        sample = []
        for _ in range(k):
            prob = random.random()
            if non_null:
                prob -= sum(v for sv, v in self.items() if sv.n == 0)
            for sv, v in self.items():
                if non_null and sv.n == 0:
                    continue
                if prob < v:
                    if k == 1:
                        return sv
                    sample.append(sv)
                    break
                prob -= v
        return sample

    def pdisplay(self, output_format="text", n_simplify=True, precision=1e-6, max_v=None, sort=True):
        if sort:
            the_keys = sorted(self.keys(), key=lambda a: -self[a])
        else:
            the_keys = list(self.keys())
        if max_v is not None:
            the_keys = the_keys[:max_v]
        d = []
        for k in the_keys:
            if isinstance(self[k], sp.Expr):
                d.append([k, str(self[k])])
            else:
                d.append([k, simple_float(self[k], nsimplify=n_simplify, precision=precision)[1]])

        s_states = tabulate(d, headers=["state ", "probability"],
                            tablefmt=output_format == "text" and "pretty" or output_format)
        return s_states


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


def convert_polarized_state(state: AnnotatedBasicState,
                            use_symbolic: bool = False,
                            inverse: bool = False) -> Tuple[BasicState, Matrix]:
    r"""Convert a polarized basic state into an expanded state vector

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
                annot = state.get_photon_annotations(idx + 1)
                idx += 1
                v_hv = annot.get("P", Polarization(0)).project_eh_ev(use_symbolic)
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
