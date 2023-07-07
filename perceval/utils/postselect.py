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

from perceval.utils.statevector import BasicState

import json
import re
from typing import Callable, List


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

    _OPERATOR = {"==": int.__eq__, "<": int.__lt__, ">": int.__gt__}
    _PATTERN = re.compile(r"(\[[,0-9\s]+\]\s*)(==|<|>)\s*(\d+\b)")

    def __init__(self, str_repr: str = None):
        self._conditions = {}
        condition_count = 0
        if str_repr is not None:
            try:
                for match in self._PATTERN.finditer(str_repr):
                    indexes = tuple(json.loads(match.group(1)))
                    self._add_condition(indexes=indexes,
                                        operator=self._OPERATOR[match.group(2)],
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
            operator_str = [o for o in self._OPERATOR if self._OPERATOR[o] == operator][0]
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
