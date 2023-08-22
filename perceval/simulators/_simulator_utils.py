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

from perceval.utils import BasicState, BSDistribution, StateVector, Annotation
from perceval.components import Circuit
from copy import copy
from math import sqrt
from typing import List


def _to_bsd(sv: StateVector) -> BSDistribution:
    res = BSDistribution()
    sv_copy = copy(sv)
    sv_copy.normalize()
    for state, pa in sv_copy.items():
        state.clear_annotations()
        res[state] += abs(pa) ** 2
    return res


def _inject_annotation(sv: StateVector, annotation: Annotation) -> StateVector:
    res_sv = copy(sv)
    if str(annotation):  # len(annotation) not working on unix
        for s in res_sv:
            s.inject_annotation(annotation)
    return res_sv


def _merge_sv(sv1: StateVector, sv2: StateVector, prob_threshold: float = 0) -> StateVector:
    if not sv1:
        return sv2
    pa_threshold = sqrt(prob_threshold)
    res = StateVector()
    for s1, pa1 in sv1.items():
        for s2, pa2 in sv2.items():
            pa = pa1*pa2
            if abs(pa) > pa_threshold:
                res[s1.merge(s2)] += pa
    return res


def _annot_state_mapping(bs_with_annots: BasicState):
    bs_list = bs_with_annots.separate_state(keep_annotations=True)
    mapping = {}
    for bs in bs_list:
        if bs.n == 0:
            mapping[Annotation()] = bs
            continue
        annot = bs.get_photon_annotation(0)
        bs.clear_annotations()
        mapping[annot] = bs
    return mapping


def _retrieve_mode_count(component_list: List) -> int:
    return max([m for r in component_list for m in r[0]]) + 1


def _unitary_components_to_circuit(component_list: List, m: int = 0):
    if not m:
        m = _retrieve_mode_count(component_list)
    circuit = Circuit(m)
    for r, c in component_list:
        circuit.add(r, c)
    return circuit
