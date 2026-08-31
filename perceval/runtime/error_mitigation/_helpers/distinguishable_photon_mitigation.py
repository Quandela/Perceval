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

from perceval.components import DetectionType
from perceval.utils import FockState, BSDistribution


def filter_extra_photons(dist: BSDistribution, n: int) -> BSDistribution:
    """Filter out states with more than `n` photons.
    """
    extracted = BSDistribution()
    for state, prob in dist.items():
        if state.n <= n:
            extracted.add(state, prob)

    return extracted


def apply_detection_filter(distribution: BSDistribution, pnr_per_mode: list[int | None], detection_type: DetectionType) -> BSDistribution:
    """Apply a detection pattern to a set of results matching the PNR.
    """
    if detection_type == DetectionType.PNR:
        return distribution

    elif detection_type == DetectionType.Threshold:
        return distribution.photon_threshold_simplification(1)

    assert len(pnr_per_mode) == distribution.m
    detected = BSDistribution()
    for state, prob in distribution.items():
        detected.add(
            FockState([
                min(count, pnr_per_mode[mode] if pnr_per_mode[mode] is not None else count)
                for mode, count in enumerate(state)
            ]),
            prob,
        )

    return detected


def generate_obb_partition(input_state: FockState, order: int):
    """
    Generate one-bad-basis partitions for a given OBB order.
    Yields the cells, and the multiplicity for each of them (i.e. the number of times they should be accounted for)
    """
    order = min(order, input_state.n)
    if order == 0:
        yield [input_state], 1
        return

    modes = len(input_state)
    non_empty_modes = [mode for mode, count in enumerate(input_state) if count > 0]

    single_fs = {mode: FockState([1 if m == mode else 0 for m in range(modes)]) for mode in non_empty_modes}

    if order == input_state.n - 1:
        # Merge all cells into one, as they are all identical with only 1-photon states
        cell = [single_fs[input_state.photon2mode(i)] for i in range(input_state.n)]
        yield cell, input_state.n
        return

    for positions in combinations_with_replacement(non_empty_modes, order):
        counts = Counter(positions)
        if any(input_state[mode] < count for mode, count in counts.items()):
            continue

        remaining = list(input_state)
        multiplicity = 1
        cell = []
        for mode, count in counts.items():
            remaining[mode] -= count
            multiplicity *= comb(input_state[mode], count)

            for _ in range(count):
                cell.append(single_fs[mode])

        if order != input_state.n:
            cell.append(FockState(remaining))

        yield cell, multiplicity


def generate_obb_states(input_state: FockState, order: int) -> list[FockState]:
    """
    Generate all input states needed by the OBB corrections up to the given order.
    Every state is inserted exactly once in the resulting list
    """
    states = [input_state]  # order = 0

    order = min(order, input_state.n)
    if order == 0:
        return states

    m = input_state.m
    non_empty_modes = [mode for mode, count in enumerate(input_state) if count > 0]

    for mode in non_empty_modes:  # order = n - 1, but these states are also needed for any order > 0
        state = [0] * m
        state[mode] = 1
        states.append(FockState(state))

    for current_order in range(1, min(order + 1, input_state.n - 1)):

        for positions in combinations_with_replacement(non_empty_modes, current_order):
            counts = Counter(positions)
            if any(input_state[mode] < count for mode, count in counts.items()):
                continue

            remaining = list(input_state)
            for mode, count in counts.items():
                remaining[mode] -= count

            states.append(FockState(remaining))

    # order n corresponds to the empty state, which is never included

    return states
