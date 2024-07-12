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

from .statevector import BasicState, BSDistribution, StateVector

import json
import re
from typing import Callable, List, Tuple


class PostSelect:
    """PostSelect is a callable post-selection intended to filter out unwanted output states. It is designed to be a
    user-friendly description of any post-selection logic.

    :param str_repr: string representation of the selection logic. The format is: "cond_1 & cond_2 & ... & cond_n"
        where cond_i is "[mode list] <operator> <photon count>" (supported operators are ==, > and <)

    Example:

    >>> ps = PostSelect("[0,1] == 1 & [2] > 1") # Means "I want exactly one photon in mode 0 or 1, and at least one photon in mode 2"
    >>> ps = PostSelect().eq([0,1], 1).gt(2, 1) # Same as above
    >>> print(ps(BasicState([0, 1, 1])))
    True
    >>> print(ps(BasicState([1, 1, 1])))
    False
    """

    _OPERATORS = {"==": int.__eq__,
                  "<": int.__lt__,
                  ">": int.__gt__,
                  ">=": int.__ge__,
                  "<=": int.__le__}

    # Regexp explanations:
    # first group: index(es) of modes between '[]' and separated with ','
    # second group: operator (listed in _OPERATORS). We suppose there is no digit character in operators
    # third group: number of photons
    _PATTERN = re.compile(r"(\[[,0-9\s]+\]\s*)([^\d]*)\s*(\d+\b)")

    def __init__(self, str_repr: str = None):
        self._conditions = {}
        condition_count = 0
        if str_repr:
            try:
                for match in self._PATTERN.finditer(str_repr):
                    indexes = tuple(json.loads(match.group(1)))

                    operator = match.group(2).strip()
                    if operator not in self._OPERATORS:
                        raise KeyError(f"Unsupported operator: {operator}")

                    self._add_condition(indexes=indexes,
                                        operator=self._OPERATORS[operator],
                                        value=int(match.group(3)))
                    condition_count += 1

            except json.decoder.JSONDecodeError as e:
                raise RuntimeError(f"Could not interpret input string '{str_repr}': {e}")
            if condition_count != str_repr.count("&") + 1:
                raise RuntimeError(f"Could not interpret input string '{str_repr}': Invalid format")

    def eq(self, indexes, value: int):
        """Create a new "equals"     condition for the current PostSelect instance"""
        self._add_condition(indexes, int.__eq__, value)
        return self

    def gt(self, indexes, value: int):
        """Create a new "greater than" condition for the current PostSelect instance"""
        self._add_condition(indexes, int.__gt__, value)
        return self

    def lt(self, indexes, value: int):
        """Create a new "lower than" condition for the current PostSelect instance"""
        self._add_condition(indexes, int.__lt__, value)
        return self

    def ge(self, indexes, value: int):
        """Create a new "greater or equal than" condition for the current PostSelect instance"""
        self._add_condition(indexes, int.__ge__, value)
        return self

    def le(self, indexes, value: int):
        """Create a new "lower or equal than" condition for the current PostSelect instance"""
        self._add_condition(indexes, int.__le__, value)
        return self

    def _add_condition(self, indexes, operator: Callable, value: int):
        indexes = (indexes,) if isinstance(indexes, int) else tuple(indexes)
        if operator not in self._conditions:
            self._conditions[operator] = []
        self._conditions[operator].append((indexes, value))

    def __call__(self, state: BasicState) -> bool:
        """PostSelect is callable, with a `post_select(BasicState) -> bool` signature.
        Returns `True` if the input state validates all conditions, returns `False` otherwise.
        """
        for operator, cond in self._conditions.items():
            for indexes, value in cond:
                s = 0
                for i in indexes:
                    s += state[i]
                if not operator(s, value):
                    return False
        return True

    def __repr__(self):
        strlist = []
        for operator, cond in self._conditions.items():
            operator_str = [o for o in self._OPERATORS if self._OPERATORS[o] == operator][0]
            for indexes, value in cond:
                strlist.append(f"{list(indexes)}{operator_str}{value}")
        return "&".join(strlist)

    def __eq__(self, other):
        return self._conditions == other._conditions

    @property
    def has_condition(self) -> bool:
        """Returns True if at least one condition is defined"""
        return len(self._conditions) > 0

    def clear(self):
        """Clear all existing conditions"""
        self._conditions.clear()

    def apply_permutation(self, perm_vector: List[int], first_mode: int = 0):
        """
        Apply a given permutation on the conditions.

        :param perm_vector: Permutation vector to apply (as returned by PERM.perm_vector)
        :param first_mode: First mode of the permutation to apply (default 0)
        :return: A PostSelect with the permutation applied
        """
        output = PostSelect()
        for operator, cond in self._conditions.items():
            output._conditions[operator] = []
            for (indexes, value) in cond:
                new_indexes = []
                for i in indexes:
                    if i < first_mode or i >= first_mode + len(perm_vector):
                        new_indexes.append(i)
                    else:
                        new_indexes.append(first_mode + perm_vector[i - first_mode])
                output._conditions[operator].append((tuple(new_indexes), value))
        return output

    def shift_modes(self, shift: int):
        """
        Shift all mode indexes inside this instance

        :param shift: Value to shift all mode indexes with
        """
        for operator, cond in self._conditions.items():
            for c, (indexes, value) in enumerate(cond):
                assert min(indexes) + shift >= 0, f"A shift of {shift} would lead to negative mode# on {self}"
                new_indexes = tuple(i + shift for i in indexes)
                cond[c] = (new_indexes, value)

    def can_compose_with(self, modes: List[int]) -> bool:
        """
        Check if all conditions are compatible with a composition on given modes

        :param modes: Modes used in the composition
        :return: `True` if the composition is allowed without mixing conditions, `False` otherwise
        """
        m_set = set(modes)
        for _, cond in self._conditions.items():
            for (indexes, _) in cond:
                i_set = set(indexes)
                if i_set.intersection(m_set) and not i_set.issuperset(m_set):
                    return False
        return True

    def merge(self, other):
        """
        Merge with other PostSelect. Updates the current instance.

        :param other: Another PostSelect instance
        """
        for operator, cond in other._conditions.items():
            for indexes, value in cond:
                self._add_condition(indexes, operator, value)


def postselect_independent(ps1: PostSelect, ps2: PostSelect) -> bool:
    ps1_mode_set = set()
    for _, cond in ps1._conditions.items():
        for (indexes, _) in cond:
            for i in indexes:
                ps1_mode_set.add(i)
    for _, cond in ps2._conditions.items():
        for (indexes, _) in cond:
            for i in indexes:
                if i in ps1_mode_set:
                    return False
    return True


def post_select_distribution(
        bsd: BSDistribution,
        postselect: PostSelect,
        heralds: dict = None,
        keep_heralds: bool = True) -> Tuple[BSDistribution, float]:
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
        keep_heralds: bool = True) -> Tuple[StateVector, float]:
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
            result += ampli*state
        else:
            logical_perf -= abs(ampli)**2
    result.normalize()
    return result, logical_perf
