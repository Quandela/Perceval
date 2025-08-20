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
import copy
from collections import defaultdict
from math import sqrt
from multipledispatch import dispatch
from typing import Iterable

from perceval.utils import FockState, NoisyFockState, AnnotatedFockState, BSDistribution, StateVector, Annotation, SVDistribution
from perceval.components import Circuit


def _to_bsd(sv: StateVector) -> BSDistribution:
    res = BSDistribution()
    for state, pa in sv.unnormalized_iterator():
        if not isinstance(state, FockState):
            state = state.clear_annotations()
        res.add(state, abs(pa) ** 2)
    return res


@dispatch(StateVector, Annotation)
def _inject_annotation(sv: StateVector, annotation: Annotation) -> StateVector:
    if isinstance(annotation, int):
        res_sv = StateVector()
        for s, pa in sv.unnormalized_iterator():
            s = NoisyFockState(s, [annotation]*s.n)
            res_sv += pa * s
        return res_sv
    if len(annotation):
        res_sv = StateVector()
        for s, pa in sv.unnormalized_iterator():
            s = copy.copy(s)
            s.inject_annotation(annotation)
            res_sv += pa * s
        return res_sv
    return sv


@dispatch(StateVector, int)
def _inject_annotation(sv: StateVector, annotation: int) -> StateVector:
    res_sv = StateVector()
    for s, pa in sv.unnormalized_iterator():
        s = NoisyFockState(s, [annotation]*s.n)
        res_sv += pa * s
    return res_sv


def _merge_sv(sv1: StateVector, sv2: StateVector, prob_threshold: float = 0) -> StateVector:
    if not sv1:
        return sv2
    pa_threshold = sqrt(prob_threshold)
    res = StateVector()
    for s1, pa1 in sv1.unnormalized_iterator():
        for s2, pa2 in sv2.unnormalized_iterator():
            pa = pa1 * pa2
            if abs(pa) > pa_threshold:
                res += s1.merge(s2)*pa
    return res


@dispatch(FockState)
def _annot_state_mapping(bs_with_annots: FockState):
    return {Annotation(): bs_with_annots}


@dispatch(NoisyFockState)
def _annot_state_mapping(bs_with_annots: NoisyFockState):
    return bs_with_annots.split_state()


@dispatch(AnnotatedFockState)
def _annot_state_mapping(bs_with_annots: AnnotatedFockState): # TODO if needed + separate_state() does not exist for AnnotatedFockState anymore
    return {bs_with_annots.get_photon_annotation(0): bs_with_annots.clear_annotations()}


def _retrieve_mode_count(component_list: list) -> int:
    return max([m for r in component_list for m in r[0]]) + 1


def _unitary_components_to_circuit(component_list: list, m: int = 0):
    if not m:
        m = _retrieve_mode_count(component_list)
    circuit = Circuit(m)
    for r, c in component_list:
        circuit.add(r, c)
    return circuit


def _split_by_photon_count(sv: StateVector) -> SVDistribution:
    """
    Split a state vector into a SVDistribution such that each key of the SVD corresponds to one photon count
    """
    counter = defaultdict(lambda: [StateVector(), 0])  # State and prob
    for state, pa in sv:
        counter[state.n][0] += pa * state
        counter[state.n][1] += abs(pa) ** 2

    res = SVDistribution()
    for (state, prob) in counter.values():
        state.normalize()
        res[state] = prob

    return res

def _list_merge(distributions: list[Iterable[tuple[FockState, float]]], prob_threshold: float = 0) -> dict[FockState, float]:
    # Note: the iterable must reset each time we call it
    if len(distributions) == 0:
        return dict()

    res = defaultdict(float)

    def _inner_tensor_product(dist: list[Iterable[tuple[FockState, float]]], current_state: FockState, current_prob: float):
        if len(dist) == 0:
            res[current_state] += current_prob
            return

        for bs, p in dist[0]:
            prob = current_prob * p
            if prob < prob_threshold:
                continue
            _inner_tensor_product(dist[1:], current_state.merge(bs), prob)

    start_state = FockState(next(iter(distributions[0]))[0].m)
    _inner_tensor_product(distributions, start_state, 1)
    return res
