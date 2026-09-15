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

from typing import Any

from perceval.components import Experiment
from perceval.utils import (
    AnnotatedFockState, FockState, NoisyFockState, StateVector,
    SVDistribution, BasicState
)
from perceval.utils.logging import channel, get_logger


class ConstraintChecker:
    """Helper class to check constraints"""

    class EmptyType:
        pass

    _TAG_MAP: dict[str, type] = {
        "FS": FockState,
        "NFS": NoisyFockState,
        "AFS": AnnotatedFockState,
        "SV": StateVector,
        "SVD": SVDistribution,
    }

    _COUNT_CONSTRAINTS = {
        "max_mode_count", "min_mode_count", "max_photon_count", "min_photon_count"
    }
    _KNOWN_CONSTRAINTS = _COUNT_CONSTRAINTS | {"accepted_state_kinds", "support_multi_photon", "max_sample_count"}

    @staticmethod
    def check_valid_constraints(constraints: dict[str, Any]) -> None:
        """
        May be used by a computer to check that the constraints are correctly written
        Unknown fields are warned about.
        Known fields are checked for correctness, and raise an error if not.
        """
        if not isinstance(constraints, dict):
            raise TypeError("Constraints must be provided as a dictionary")

        for field in constraints.keys() - ConstraintChecker._KNOWN_CONSTRAINTS:
            get_logger().warn(f"Unknown constraint field: {field}", channel.user)

        for field in ConstraintChecker._COUNT_CONSTRAINTS & constraints.keys():
            value = constraints[field]
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"Constraint '{field}' must be an integer")
            if value < 0:
                raise ValueError(f"Constraint '{field}' must be non-negative")

        for minimum, maximum in (("min_mode_count", "max_mode_count"),
                                 ("min_photon_count", "max_photon_count")):
            if minimum in constraints and maximum in constraints \
                    and constraints[minimum] > constraints[maximum]:
                raise ValueError(f"Constraint '{minimum}' cannot be greater than '{maximum}'")

        if "support_multi_photon" in constraints \
                and not isinstance(constraints["support_multi_photon"], bool):
            raise TypeError("Constraint 'support_multi_photon' must be a boolean")

        if "accepted_state_kinds" in constraints:
            state_kinds = constraints["accepted_state_kinds"]
            if not isinstance(state_kinds, list):
                raise TypeError("Constraint 'accepted_state_kinds' must be a list")
            if not state_kinds:
                raise ValueError("Constraint 'accepted_state_kinds' cannot be empty")
            if any(not isinstance(kind, str) for kind in state_kinds):
                raise TypeError("Every accepted state kind must be a string")

            unknown_kinds = set(state_kinds) - ConstraintChecker._TAG_MAP.keys()
            if unknown_kinds:
                raise ValueError(f"Unknown accepted state kind(s): {', '.join(sorted(unknown_kinds))}")
            if len(state_kinds) != len(set(state_kinds)):
                raise ValueError("Constraint 'accepted_state_kinds' cannot contain duplicates")
            if not {"FS", "NFS", "AFS"}.intersection(state_kinds):
                raise ValueError("Constraint 'accepted_state_kinds' must contain a basic state kind "
                                 "('FS', 'NFS', or 'AFS')")

    @staticmethod
    def _state_kind(state: BasicState) -> str | None:
        """Return the constraint tag for a concrete basic-state type."""
        state_type = type(state)
        return next((tag for tag in ("FS", "NFS", "AFS")
                     if ConstraintChecker._TAG_MAP[tag] is state_type), None)

    @staticmethod
    def _unsupported_input(input_state, accepted_state_types: tuple[type, ...]) -> RuntimeError:
        return RuntimeError(f"Unsupported input state {input_state} for accepted state kinds "
                            f"{[t.__name__ for t in accepted_state_types if t is not ConstraintChecker.EmptyType]}")

    @staticmethod
    def verify_input(constraints: dict[str, Any], experiment: Experiment) -> None:
        """
        Raises an error if the given input state is invalid in regard to the constraints.

        :param constraints: The constraints of a computer.
        :param experiment: The user Experiment.
        """
        input_state = experiment.input_state
        if input_state is None:
            raise ValueError("The experiment has no input_state (call `with_input()`)")

        if constraints:
            # Checks on state

            # Rules:
            # SV: Accept superposition of mentioned states (e.g. ["FS", "SV"] accepts superposition of FS but not of NFS)
            # SVD: accept len(SVD) > 1
            # NFS or SV without SVD imposes len(SVD) == 1.
            # FS or AFS alone forbid the SVD type.
            # SVD without SV accepts only mixtures of non-superposed states

            accepted_types = tuple(ConstraintChecker._TAG_MAP.get(tag, ConstraintChecker.EmptyType)
                                   for tag in constraints.get("accepted_state_kinds", ["FS"]))
            accepted_types_cp = accepted_types


            if isinstance(input_state, BasicState):
                if not isinstance(input_state, accepted_types):
                    raise ConstraintChecker._unsupported_input(input_state, accepted_types)
            elif isinstance(input_state, SVDistribution):
                # An SVD is an implementation detail when Experiment.with_input receives
                # a StateVector or NoisyFockState.  Without explicit SVD support, that
                # single-entry wrapper is therefore accepted only for NFS/SV inputs.
                if SVDistribution not in accepted_types:
                    if len(input_state) != 1:
                        raise ConstraintChecker._unsupported_input(input_state, accepted_types)
                    if StateVector not in accepted_types:
                        if NoisyFockState in accepted_types:
                            accepted_types = (NoisyFockState,)
                        else:
                            raise ConstraintChecker._unsupported_input(input_state, accepted_types)

                for state_vector in input_state.keys():
                    if len(state_vector) > 1 and StateVector not in accepted_types:
                        raise ConstraintChecker._unsupported_input(input_state, accepted_types_cp)
                    for state in state_vector.keys():
                        if not isinstance(state, accepted_types_cp):
                            raise ConstraintChecker._unsupported_input(input_state, accepted_types_cp)
            else:
                raise ConstraintChecker._unsupported_input(input_state, accepted_types)

            # Other checks
            n_photons = input_state.n if isinstance(input_state, BasicState) else input_state.n_max
            if 'max_photon_count' in constraints and n_photons > constraints['max_photon_count']:
                n_heralds = sum(experiment.in_heralds.values())
                raise RuntimeError(
                    f"Too many photons in input state ({n_photons - n_heralds} + {n_heralds} heralds > {constraints['max_photon_count']})")
            if 'min_photon_count' in constraints and n_photons < constraints['min_photon_count']:
                raise RuntimeError(
                    f"Not enough photons in input state ({n_photons} < {constraints['min_photon_count']})")
            if 'support_multi_photon' in constraints and not constraints['support_multi_photon']:
                if isinstance(input_state, BasicState):
                    if not all(mode_photon_cnt <= 1 for mode_photon_cnt in input_state):
                        raise RuntimeError(f"Input state ({input_state}) is not permitted."
                                           " QPU/QPU simulators doesn't accept more than 1 photon per mode")
                else:
                    # SVD
                    for sv in input_state.keys():
                        for state in sv.keys():
                            if not all(mode_photon_cnt <= 1 for mode_photon_cnt in state):
                                raise RuntimeError(f"Input state ({state}) is not permitted."
                                                   " QPU/QPU simulators doesn't accept more than 1 photon per mode")

    @staticmethod
    def verify_circuit(constraints: dict[str, Any], experiment: Experiment):
        """
        Raises an error if the given experiment circuit is invalid in regard to the constraints.

        :param constraints:
        :param experiment:
        :return:
        """
        m = experiment.circuit_size
        if 'max_mode_count' in constraints and m > constraints['max_mode_count']:
            raise RuntimeError(f"Circuit too big ({m} modes > {constraints['max_mode_count']})")
        if 'min_mode_count' in constraints and m < constraints['min_mode_count']:
            raise RuntimeError(f"Circuit too small ({m} < {constraints['min_mode_count']})")

        # TODO: Check that the component matches what the platform can do
        # if new_component is not None:
        #     if isinstance(new_component, Experiment):
        #         if not new_component.is_unitary:
        #             raise RuntimeError('Cannot compose a RemoteProcessor with a processor containing non linear components')
        #         if new_component.has_feedforward:
        #             raise RuntimeError('Cannot compose a RemoteProcessor with a processor containing feed-forward')
        #
        #     elif not isinstance(new_component, IDetector) and not isinstance(new_component, ACircuit):
        #         raise NotImplementedError("Non linear components not implemented for RemoteProcessors")

    @staticmethod
    def verify_experiment(constraints: dict[str, Any], experiment: Experiment):
        """
        Raises an error if the given experiment is invalid in regard to the constraints.

        :param constraints: The constraints of a computer.
        :param experiment: The user experiment
        """
        # Give the whole specs to access the architecture if needed?
        ConstraintChecker.verify_input(constraints, experiment)
        ConstraintChecker.verify_circuit(constraints, experiment)
