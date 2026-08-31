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
from unittest.mock import patch

import pytest

import perceval as pcvl
from perceval import Experiment, BS, SimulatedComputer, DistinguishablePhotonMitigation, FockState, NoiseModel, Computation, \
    tvd_dist
from tests._test_utils import assert_bsd_close, assert_unordered_lists_equal
from perceval.runtime.error_mitigation._helpers.distinguishable_photon_mitigation import generate_obb_partition, generate_obb_states


def test_state_generation():
    input_state = FockState(3 * [1])
    order = 0

    all_cells = list(generate_obb_partition(input_state, order))
    assert len(all_cells) == 1
    assert all_cells[0][0] == [input_state]
    assert all_cells[0][1] == 1

    order = 1
    all_cells = list(generate_obb_partition(input_state, order))
    assert len(all_cells) == 3
    expected = [([FockState([1, 0, 0]), FockState([0, 1, 1])], 1),
                ([FockState([0, 1, 0]), FockState([1, 0, 1])], 1),
                ([FockState([0, 0, 1]), FockState([1, 1, 0])], 1)]
    assert_unordered_lists_equal(all_cells, expected)

    order = 2
    all_cells = list(generate_obb_partition(input_state, order))
    expected = [([FockState([1, 0, 0]), FockState([0, 1, 0]), FockState([0, 0, 1])], 3)]
    assert all_cells == expected

    order = 3
    all_cells = list(generate_obb_partition(input_state, order))
    expected = [([FockState([1, 0, 0]), FockState([0, 1, 0]), FockState([0, 0, 1])], 1)]
    assert all_cells == expected


    # Test holes in the state
    input_state = FockState(3 * [1, 0])

    order = 1
    all_cells = list(generate_obb_partition(input_state, order))
    assert len(all_cells) == 3
    expected = [([FockState([1, 0, 0, 0, 0, 0]), FockState([0, 0, 1, 0, 1, 0])], 1),
                ([FockState([0, 0, 1, 0, 0, 0]), FockState([1, 0, 0, 0, 1, 0])], 1),
                ([FockState([0, 0, 0, 0, 1, 0]), FockState([1, 0, 1, 0, 0, 0])], 1)]
    assert_unordered_lists_equal(all_cells, expected)


    # Test with multiple photons in an input mode
    input_state = FockState([2, 1])
    order = 1
    all_cells = list(generate_obb_partition(input_state, order))
    assert len(all_cells) == 2
    expected = [([FockState([1, 0]), FockState([1, 1])], 2),
                ([FockState([0, 1]), FockState([2, 0])], 1)]
    assert_unordered_lists_equal(all_cells, expected)

    order = 2
    all_cells = list(generate_obb_partition(input_state, order))
    expected = [([FockState([1, 0]), FockState([1, 0]), FockState([0, 1])], 3)]
    assert all_cells == expected


@pytest.mark.parametrize("input_state, order", [(FockState(3 * [1]), 0),
                                                (FockState(3 * [1]), 1),
                                                (FockState(3 * [1]), 2),
                                                (FockState(3 * [1]), 3),
                                                (FockState(5 * [1, 0]), 0),
                                                (FockState(5 * [1, 0]), 2),
                                                (FockState(5 * [1, 0]), 5),
                                                (FockState([2, 1, 0]), 0),
                                                (FockState([2, 1, 0]), 1),
                                                (FockState([2, 1, 0]), 2),
                                                (FockState([2, 1, 0]), 3),
                                                ])
def test_state_generation_equivalence(input_state, order):
    state_set = set((st for i in range(order + 1) for state in generate_obb_partition(input_state, i) for st in state[0]))
    state_list = generate_obb_states(input_state, order)

    assert len(state_set) == len(state_list)
    assert set(state_list) == state_set


def test_overhead():
    assert DistinguishablePhotonMitigation(2).overhead(FockState([1, 1])) == 3


def test_basic_hom_mitigation():
    e = Experiment(BS())
    e.with_input(FockState([1, 1]))
    e.min_detected_photons_filter(2)

    c = SimulatedComputer("SLOS")
    computation = Computation(c.get_command("probs"), e)

    perfect_res = c.execute(computation)

    c.noise = NoiseModel(indistinguishability=0.8)
    unmitigated_res = c.execute(computation)

    with pytest.raises(AssertionError):
        assert_bsd_close(unmitigated_res["results"], perfect_res["results"])

    c.mitigations = [DistinguishablePhotonMitigation(2)]
    corrected_res = c.execute(computation)

    # In the HOM experiment case, we can perfectly correct the errors
    assert_bsd_close(corrected_res["results"], perfect_res["results"])


def test_g2_mitigation():
    e = Experiment(BS())
    e.with_input(FockState([1, 1]))
    e.min_detected_photons_filter(2)

    c = SimulatedComputer("SLOS")
    computation = Computation(c.get_command("probs"), e)

    perfect_res = c.execute(computation)

    c.noise = NoiseModel(g2=0.05)
    unmitigated_res = c.execute(computation)

    with pytest.raises(AssertionError):
        assert_bsd_close(unmitigated_res["results"], perfect_res["results"])

    c.mitigations = [DistinguishablePhotonMitigation(2)]
    corrected_res = c.execute(computation)

    # In the HOM experiment case, we can perfectly correct the errors
    assert_bsd_close(corrected_res["results"], perfect_res["results"])


@patch.object(pcvl.utils.logging.ExqaliburLogger, "warn")  # Suppress the warning in tvd_dist()
def test_full_noise(mock_warn):
    e = Experiment(BS())
    e.with_input(FockState([1, 1]))
    e.min_detected_photons_filter(1)

    c = SimulatedComputer("SLOS")
    computation = Computation(c.get_command("probs"), e)

    c.noise = NoiseModel(transmittance=0.06)
    perfect_res = c.execute(computation)

    c.noise = NoiseModel(indistinguishability= 0.8, g2=0.05, transmittance=0.06)
    non_corrected_res = c.execute(computation)

    c.mitigations = [DistinguishablePhotonMitigation(2)]
    corrected_res = c.execute(computation)

    tvd_non_corrected = tvd_dist(perfect_res["results"], non_corrected_res["results"])
    tvd_corrected = tvd_dist(perfect_res["results"], corrected_res["results"])

    assert tvd_corrected < tvd_non_corrected
