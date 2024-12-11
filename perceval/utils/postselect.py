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

from perceval.utils.logging import deprecated

import exqalibur as xq

from .statevector import BSDistribution, StateVector, BasicState


class PostSelect(xq.PostSelect):
    """PostSelect is a callable basic state predicate intended to filter out unwanted basic states. It is designed to be
    a user-friendly description of any post-selection logic.

    :param expression: PostSelect string representation of the post-selection logic.
    :type expression: str

    PostSelect Syntax
    =================

    Condition
    ---------
        Condition syntax is `[mode list] <operator> <photon count>`

        PostSelect expression is always composed of at least one condition.

        - Mode list:
            - string that represented a list[int]
            - should be a set of positive integer

        - Operator:
            - string that represent an operator.
            - Operator within a condition can be:
                - Equal: "=="
                - Greater than: ">"
                - Greater or equal to: ">="
                - Less than: "<"
                - Less or equal to: "<="

        - Photon count:
            - string that represent a non-negative integer

    Logic operators
    ---------------
        Within a PostSelect expression, several conditions can be composed with operators, grouped (with parenthesis)
        and even nested.

        Condition(s) composed with an operator will is consider as a condition group.

        Available logic operators are:
            - AND:
                - can compose 2 or more condition groups
                - possible string representation:
                    - "AND"
                    - "and"
                    - "&" (default serialization string representation)
            - OR:
                - can compose 2 or more condition groups
                - possible string representation:
                    - "OR"
                    - "or"
                    - "|" (default serialization string representation)
            - XOR:
                - can compose 2 or more condition groups
                - possible string representation:
                    - "XOR"
                    - "xor"
                    - "^" (default serialization string representation)
            - NOT:
                - can be used in front of a condition group
                - possible string representation:
                    - "NOT"
                    - "not"
                    - "!" (default serialization string representation)

        Different operators cannot be used within the same condition group, parenthesis are necessary in order to
        explicit resolution order.

    Examples
    ========

    >>> ps = PostSelect("[0,1] == 1 & [2] > 1") # Means "I want exactly one photon in mode 0 or 1, and at least one photon in mode 2"
    >>> print(ps(BasicState([0, 1, 2])))
    True
    >>> print(ps(BasicState([0, 1, 0])))
    False
    >>> print(ps(BasicState([1, 1, 2])))
    False

    >>> ps = PostSelect("([0,1] == 1 & [2] > 1) | [2] == 0") # Means "I want either exactly one photon in mode 0 or 1, and at least one photon in mode 2, or no photon in mode 2"
    >>> print(ps(BasicState([0, 1, 1])))
    False
    >>> print(ps(BasicState([0, 1, 0])))
    True
    >>> print(ps(BasicState([1, 1, 2])))
    False
    """

    def __call__(self, state: BasicState) -> bool:
        """Return whether a state validates the defined post-selection logic.

        :param state: Basic state to post select
        :return: Returns `True` if the input state validates the defined post-selection logic, returns `False` otherwise.
        """
        return super().__call__(state)

    @property
    def has_condition(self) -> bool:
        """Returns True if at least one condition is defined"""
        return super().has_condition

    def clear(self) -> None:
        """Clear all existing conditions"""
        super().clear()

    def shift_modes(self, shift: int):
        """
        Shift all mode indexes on all conditions. Updates the current instance.

        :param shift: Value to shift all mode indexes with
        """
        super().shift_modes(shift)

    def apply_permutation(self, perm_vector: list[int], first_mode: int = 0):
        """
        Apply a given permutation on the conditions. Updates the current instance.

        :param perm_vector: Permutation vector to apply (as returned by PERM.perm_vector)
        :param first_mode: First mode to apply the permutation on (default 0)
        """
        super().apply_permutation(perm_vector, first_mode)

    def can_compose_with(self, modes: list[int]) -> bool:
        """
        Check if all conditions are compatible with a composition on given modes.

        Compatible means that modes list is either a subset, or doesn't intersect with any mode list of any conditions.

        :param modes: Mode list to check compatibility
        :return: `True` if the mode list is compatible, `False` otherwise
        """
        return super().can_compose_with(modes)

    def is_independent_with(self, other: PostSelect) -> bool:
        """
        Check if other PostSelect instance is independent with current one.

        Independent means that current and other instances does not share any modes within their conditions.

        :param other: Another PostSelect instance
        :return: `True` if current and other instances are independent, `False` otherwise
        """
        return super().is_independent_with(other)

    def merge(self, other: PostSelect):
        """
        Merge other PostSelect with an AND operator. Updates the current instance.

        :param other: Another PostSelect instance
        """
        super().merge(other)


@deprecated(version="0.12.0", reason="Use instead PostSelect class method `is_independent_with`")
def postselect_independent(ps1: PostSelect, ps2: PostSelect) -> bool:
    """ Check if two PostSelect instances are independent.

    :param ps1: First post selection
    :param ps2: Second post selection
    :return: `True` if PostSelect instances are independent, `False` otherwise
    """
    return ps1.is_independent_with(ps2)


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
    """Post select a state vector

    :param sv: State Vector to post select
    :param postselect: post selection to apply
    :param heralds: heralds to apply, defaults to None
    :param keep_heralds: Keep heralded modes in the BSDistribution (heralded modes will be removed from Fock states),
                         defaults to True
    :return:  A tuple containing the post-selected StateVector and logical performance
    """
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
