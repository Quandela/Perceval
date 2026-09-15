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

import pytest

from perceval.components import Experiment
from perceval.runtime.contraint_checker import ConstraintChecker
from perceval.utils import BasicState, StateVector, SVDistribution


def _experiment(input_state):
    experiment = Experiment(2)
    experiment.with_input(input_state)
    return experiment


@pytest.mark.parametrize("constraints", [
    {},
    {"min_mode_count": 1, "max_mode_count": 2,
     "min_photon_count": 0, "max_photon_count": 2,
     "support_multi_photon": False,
     "accepted_state_kinds": ["FS", "SV", "SVD"]},
])
def test_valid_constraints(constraints):
    ConstraintChecker.check_valid_constraints(constraints)


@pytest.mark.parametrize("constraints, exception", [
    ({"max_mode_count": 1.5}, TypeError),
    ({"min_photon_count": -1}, ValueError),
    ({"support_multi_photon": 1}, TypeError),
    ({"accepted_state_kinds": "FS"}, TypeError),
    ({"accepted_state_kinds": ["unknown"]}, ValueError),
    ({"min_mode_count": 3, "max_mode_count": 2}, ValueError),
])
def test_invalid_constraints(constraints, exception):
    with pytest.raises(exception):
        ConstraintChecker.check_valid_constraints(constraints)


def test_state_vector_rules():
    fs_superposition = StateVector([1, 0]) + StateVector([0, 1])
    experiment = _experiment(fs_superposition)

    ConstraintChecker.verify_input({"accepted_state_kinds": ["FS", "SV"]}, experiment)
    with pytest.raises(RuntimeError, match="Unsupported input state"):
        ConstraintChecker.verify_input({"accepted_state_kinds": ["FS", "SVD"]}, experiment)
    with pytest.raises(RuntimeError, match="Unsupported input state"):
        ConstraintChecker.verify_input({"accepted_state_kinds": ["FS"]}, experiment)
    with pytest.raises(RuntimeError, match="Unsupported input state"):
        ConstraintChecker.verify_input({"accepted_state_kinds": ["AFS", "SV"]}, experiment)


def test_state_vector_distribution_rules():
    mixture = SVDistribution({StateVector([1, 0]): 0.5, StateVector([0, 1]): 0.5})
    experiment = _experiment(mixture)

    ConstraintChecker.verify_input({"accepted_state_kinds": ["FS", "SVD"]}, experiment)
    with pytest.raises(RuntimeError, match="Unsupported input state"):
        ConstraintChecker.verify_input({"accepted_state_kinds": ["FS", "SV"]}, experiment)


def test_distribution_without_state_vector_rejects_superposition():
    superposition = StateVector([1, 0]) + StateVector([0, 1])
    mixture = SVDistribution({superposition: 1})

    with pytest.raises(RuntimeError, match="Unsupported input state"):
        ConstraintChecker.verify_input({"accepted_state_kinds": ["FS", "SVD"]}, _experiment(mixture))


def test_state_vector_only_accepts_single_implicit_distribution_entry():
    experiment = _experiment(StateVector(BasicState([1, 0])))
    ConstraintChecker.verify_input({"accepted_state_kinds": ["FS", "SV"]}, experiment)
    with pytest.raises(RuntimeError, match="Unsupported input state"):
        ConstraintChecker.verify_input({"accepted_state_kinds": ["FS"]}, experiment)


def test_noisy_fs():
    experiment = _experiment(BasicState("|{0},{1}>"))
    ConstraintChecker.verify_input({"accepted_state_kinds": ["NFS"]}, experiment)


def test_verify_experiment_checks_input_and_circuit():
    experiment = _experiment(BasicState([1, 0]))
    ConstraintChecker.verify_experiment({"accepted_state_kinds": ["FS"], "min_mode_count": 2}, experiment)
