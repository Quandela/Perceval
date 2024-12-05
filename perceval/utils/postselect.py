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

from copy import copy
from typing import TypeAlias, Self

from .statevector import BasicState, BSDistribution, StateVector
from exqalibur import PostSelect as ex_PostSelect
from exqalibur import LogicOperator

from perceval.utils.logging import deprecated

operands: TypeAlias = LogicOperator


class PostSelect(ex_PostSelect):
    """PostSelect is a callable post-selection intended to filter out unwanted output states. It is designed to be a
    user-friendly description of any post-selection logic.

    Init uses a string representation of the selection logic. The format is: "cond_1 OP cond_2 OP ... OP cond_n"
    where cond_i is "[mode list] <operator> <photon count>" (supported operators are ==, >, <, >=, <=).

    Logic operators between conditions can be
        - "AND", "and", "&"
        - "OR", "or", "|"
        - "XOR", "xor"
        - "NOT", "not", "!"

    Different operators can be used using parentheses to separate them.
    "and", "or" and "xor" operators can be chained without the need of added parentheses.

    Example:

    >>> ps = PostSelect("[0,1] == 1 & [2] > 1") # Means "I want exactly one photon in mode 0 or 1, and at least one photon in mode 2"
    >>> ps = PostSelect().eq([0,1], 1).gt(2, 1) # Same as above
    >>> print(ps(BasicState([0, 1, 1])))
    True
    >>> print(ps(BasicState([1, 1, 1])))
    False
    """

    @deprecated(version="0.12.0", reason="Use merge instead")
    def eq(self, indexes: int | list[int], value: int) -> Self:
        """Create a new "equals"     condition for the current PostSelect instance"""
        if isinstance(indexes, int):
            indexes = [indexes]
        self.merge(PostSelect(f"{list(indexes)} == {value}"))
        return self

    @deprecated(version="0.12.0", reason="Use merge instead")
    def gt(self, indexes, value: int) -> Self:
        """Create a new "greater than" condition for the current PostSelect instance"""
        if isinstance(indexes, int):
            indexes = [indexes]
        self.merge(PostSelect(f"{list(indexes)} > {value}"))
        return self

    @deprecated(version="0.12.0", reason="Use merge instead")
    def lt(self, indexes, value: int) -> Self:
        """Create a new "lower than" condition for the current PostSelect instance"""
        if isinstance(indexes, int):
            indexes = [indexes]
        self.merge(PostSelect(f"{list(indexes)} < {value}"))
        return self

    @deprecated(version="0.12.0", reason="Use merge instead")
    def ge(self, indexes, value: int) -> Self:
        """Create a new "greater or equal than" condition for the current PostSelect instance"""
        if isinstance(indexes, int):
            indexes = [indexes]
        self.merge(PostSelect(f"{list(indexes)} >= {value}"))
        return self

    @deprecated(version="0.12.0", reason="Use merge instead")
    def le(self, indexes, value: int) -> Self:
        """Create a new "lower or equal than" condition for the current PostSelect instance"""
        if isinstance(indexes, int):
            indexes = [indexes]
        self.merge(PostSelect(f"{list(indexes)} <= {value}"))
        return self


def postselect_independent(ps1: PostSelect, ps2: PostSelect) -> bool:
    return ps1.is_independent_with(ps2)


def post_select_distribution(
        bsd: BSDistribution,
        postselect: PostSelect,
        heralds: dict = None,
        keep_heralds: bool = True) -> tuple[BSDistribution, float]:
    if not (postselect.has_condition or heralds):
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
    result.normalize()
    return result, logical_perf


def post_select_statevector(
        sv: StateVector,
        postselect: PostSelect,
        heralds: dict = None,
        keep_heralds: bool = True) -> tuple[StateVector, float]:
    if not (postselect.has_condition or heralds):
        sv.normalize()
        return sv, 1

    if heralds is None:
        heralds = {}
    logical_perf = 1
    result = StateVector()
    for state, ampli in sv:
        heralds_ok = True
        for m, v in heralds.items():
            if state[m] != v:
                heralds_ok = False
        if heralds_ok and postselect(state):
            if not keep_heralds:
                state = state.remove_modes(list(heralds.keys()))
            result += ampli * state
        else:
            logical_perf -= abs(ampli) ** 2
    result.normalize()
    return result, logical_perf
