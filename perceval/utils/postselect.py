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

import exqalibur as xq

from .statevector import BSDistribution, StateVector
try:
    from typing import TypeAlias
except ImportError:
    from typing_extensions import TypeAlias  # Only used with python 3.9

PostSelect: TypeAlias = xq.PostSelect
# Temporary implementation of __deepcopy__ until exqalibur release
# PCVL-969
PostSelect.__deepcopy__ = lambda self, memo : self.__copy__()

def post_select_distribution(
        bsd: BSDistribution,
        postselect: PostSelect,
        heralds: dict = None,
        keep_heralds: bool = True) -> tuple[BSDistribution, float]:
    """Post select a BSDistribution

    :param bsd: BSDistribution to post select
    :param postselect: Post-selection conditions to apply
    :param heralds: Heralds to apply, defaults to None
    :param keep_heralds: Keep heralded modes in the BSDistribution (heralded modes will be removed from Fock states),
                         defaults to True
    :return: A tuple containing post-selected BSDistribution and logical performance
    """
    if not (postselect.has_condition or heralds):
        if len(bsd):
            bsd.normalize()
        return bsd, 1

    if heralds is None:
        heralds = {}
    logical_perf = 1
    result = BSDistribution()
    for state, prob in bsd.items():
        heralds_ok = True
        for m, v in heralds.items():
            if state[m] != v:
                heralds_ok = False
        if heralds_ok and postselect(state):
            if not keep_heralds:
                state = state.remove_modes(list(heralds.keys()))
            result[state] = prob
        else:
            logical_perf -= prob
    if len(result):
        result.normalize()
    return result, logical_perf


def post_select_statevector(
        sv: StateVector,
        postselect: PostSelect,
        heralds: dict = None,
        keep_heralds: bool = True) -> tuple[StateVector, float]:
    """Post select a state vector

    :param sv: State Vector to post select
    :param postselect: post selection to apply
    :param heralds: heralds to apply, defaults to None
    :param keep_heralds: Keep heralded modes in the BSDistribution (heralded modes will be removed from Fock states),
                         defaults to True
    :return:  A tuple containing the post-selected StateVector and logical performance
    """
    if not (postselect.has_condition or heralds):
        if len(sv):
            sv.normalize()
        return sv, 1

    if heralds is None:
        heralds = {}
    logical_perf = 0
    result = StateVector()
    for state, ampli in sv.unnormalized_iterator():
        heralds_ok = True
        for m, v in heralds.items():
            if state[m] != v:
                heralds_ok = False
        if heralds_ok and postselect(state):
            if not keep_heralds:
                state = state.remove_modes(list(heralds.keys()))
            result += ampli * state
            logical_perf += abs(ampli) ** 2
    if len(result):
        result.normalize()
    return result, logical_perf
