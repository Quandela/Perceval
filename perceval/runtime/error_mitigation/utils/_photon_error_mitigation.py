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

from collections import Counter
from itertools import combinations_with_replacement
from math import comb
from typing import Iterable

from perceval.utils import FockState, BSDistribution


def _extract_photon_number(dist: BSDistribution, n: int) -> BSDistribution:
    extracted = BSDistribution()
    for state, prob in dist.items():
        if state.n == n:
            extracted.add(state, prob)

    return extracted


def _apply_detection_filter(distribution: BSDistribution, pnr_per_mode: list[int]) -> BSDistribution:
    """Apply a detection pattern to a set of results matching the PNR.
    """
    if not pnr_per_mode:
        return distribution

    assert len(pnr_per_mode) == distribution.m
    detected = BSDistribution()
    for state, prob in distribution.items():
        detected.add(
            FockState([
                min(count, pnr_per_mode[mode])
                for mode, count in enumerate(state)
            ]),
            prob,
        )

    return detected


def _generate_obb_partition(input_state: FockState, order: int):
    """Generate one-bad-basis partitions for a given OBB order."""
    order = min(order, input_state.n)
    if order == 0:
        return [[input_state]]

    modes = len(input_state)
    eligible = [mode for mode, count in enumerate(input_state) if count > 0]
    partitions = []

    for positions in combinations_with_replacement(eligible, order):
        counts = Counter(positions)
        if any(input_state[mode] < count for mode, count in counts.items()):
            continue

        multiplicity = 1
        for mode, count in counts.items():
            multiplicity *= comb(input_state[mode], count)

        remaining = list(input_state)
        for mode, count in counts.items():
            remaining[mode] -= count

        cell = []
        if sum(remaining) > 0:
            cell.append(FockState(remaining))
        for mode in positions:
            state = [0] * modes
            state[mode] = 1
            cell.append(FockState(state))
        partitions.extend([cell] * multiplicity)

    return partitions


def _flatten_fock_states(obj: Iterable | FockState):
    """Helper function to flatten nested lists of FockStates in
    _generate_obb_states.
    """
    if isinstance(obj, FockState):
        yield obj
        return

    for item in obj:
        yield from _flatten_fock_states(item)


def _generate_obb_states(input_state: FockState, order: int) -> list[FockState]:
    """Generate all input states needed by the OBB corrections.
    """
    states = set()
    for current_order in range(order + 1):
        partition = _generate_obb_partition(input_state, current_order)
        states.update(_flatten_fock_states(partition))

    return sorted(states, key=tuple, reverse=True)
