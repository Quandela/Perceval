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

from collections import defaultdict

from multipledispatch import dispatch
from typing import Generator
try:
    from typing import TypeAlias
except ImportError:
    from typing_extensions import TypeAlias  # Only used with python 3.9

import exqalibur as xq


BasicState: TypeAlias = xq.FockState
FockStateIndex: TypeAlias = xq.FockStateIndex
FockStateCode: TypeAlias = xq.FockStateCode
FockStateCodeInv: TypeAlias = xq.FockStateCodeInv
BSCount: TypeAlias = xq.BSCount
BSSamples: TypeAlias = xq.BSSamples
StateVector: TypeAlias = xq.StateVector
BSDistribution: TypeAlias = xq.BSDistribution
FSIDistribution: TypeAlias = xq.FockStateIndexDistribution
FSCDistribution: TypeAlias = xq.FockStateCodeDistribution
FSCIDistribution: TypeAlias = xq.FockStateCodeInvDistribution
SVDistribution: TypeAlias = xq.SVDistribution


def allstate_array(input_state: BasicState, mask: xq.FSMask = None) -> xq.FSArray:
    m = input_state.m
    n = input_state.n
    if mask is not None:
        output_array = xq.FSArray(m, n, mask)
    else:
        output_array = xq.FSArray(m, n)

    output_array.generate()


def allstate_iterator(input_state: BasicState | StateVector, mask: xq.FSMask = None) -> Generator[xq.FockState]:
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


def tensorproduct(states: list[StateVector | BasicState | FockStateIndex | FockStateCode | FockStateCodeInv]) -> StateVector | BasicState | FockStateIndex | FockStateCode | FockStateCodeInv:
    r""" Computes states[0] * states[1] * ...
    """
    if len(states) == 1:
        return states[0]
    return tensorproduct(states[:-2] + [states[-2] * states[-1]])


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
