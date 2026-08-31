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

import math
from copy import copy, deepcopy

from perceval.components import DetectionType, get_detection_type
from perceval.utils import BSDistribution, FockState, NoiseModel, get_logger
from perceval.utils.constants import KEY_MAX_SHOTS, KEY_MAX_SAMPLES, KEY_RESULTS
from perceval.serialization import Serialization

from ..computation import Computation
from .imperfections import Imperfections, update_imperfections_from_results
from .abstract_mitigation import AbstractMitigation
from ._helpers.distinguishable_photon_mitigation import (generate_obb_states, apply_detection_filter, filter_extra_photons,
                                                         generate_obb_partition)


class DistinguishablePhotonMitigation(AbstractMitigation):
    """
    Partial distinguishability and g2 mitigation.
    Only FockState inputs are supported.
    All output states having more than the input number of photons are filtered out.

    Mitigates errors associated with noise photons errors (distinguishability and g2) by preparing jobs with fewer
    photons and recombining them through corrections based on the partial distinguishability 'orthogonal bad bits' model.

    :param order: Extent of photon error mitigation. If an integer is given,
        the correction is fixed up to ``order`` or the input photon number.
        If a dict is given, it the input photon number as key and the corresponding order as value.
    """

    def __init__(self, order: int | dict[int, int]):
        self._validate_order(order)
        self._order = order

    def overhead(self, input_state: FockState) -> int:
        """Return the number of sub-computations needed for a given input
        state based on the instance order.

        :param input_state: Provided input-state or number of photons.
        """
        input_state = self._validate_input_state(input_state)

        order = self._resolve_order(input_state.n)
        return len(generate_obb_states(input_state, order))

    def extend_computation(
        self,
        computation: Computation,
        imperfections: Imperfections
    ) -> list[Computation]:
        """Add computations for every possible sub-n photon number up to a
        given specified order of correction.
        """
        noise = imperfections.noise
        if noise.g2 == 0 and noise.indistinguishability == 1:
            return [computation]  # Nothing to mitigate here

        if noise.g2 > .5:
            raise ValueError("PhotonErrorMitigation requires g2 <= 0.5.")

        input_state = computation.experiment.input_state
        input_state = self._validate_input_state(input_state)

        resolved_order = self._resolve_order(input_state.n)

        # Note: We need the extension to be deterministic
        new_input_states = generate_obb_states(input_state, resolved_order)
        ratios = self._split_ratios(new_input_states, noise.transmittance * noise.brightness)

        samples = self._split_integer(
            computation.parameters.get(KEY_MAX_SAMPLES),
            ratios,
            KEY_MAX_SAMPLES,
        )
        shots = self._split_integer(
            computation.parameters.get(KEY_MAX_SHOTS),
            ratios,
            KEY_MAX_SHOTS,
        )

        sub_computations = []
        for i, state in enumerate(new_input_states):
            comp = self._copy_computation(
                computation,
                input_state=state,
                samples=samples[i],
                shots=shots[i],
                job_name=f"{computation.job_name} DPM {i}",
            )
            sub_computations.append(comp)

        return sub_computations

    def _parse_results(
        self,
        computation: Computation,
        results: list[dict],
        imperfections: Imperfections
    ) -> dict:
        """Mitigate distinguishability & lossy g2 contributions. Post-select
        out g2 states.

        - g2 states are post-selected out - but all states compliant with min-photon filter are kept.
        """
        if len(results) == 1:
            return results[0]

        sub_comps = self.extend_computation(computation, imperfections)

        # results[0] is most likely not the median time noise, but it is the one that carries the most information
        imperfections = update_imperfections_from_results(imperfections, results[0])
        noise = imperfections.noise

        state_idx: dict[FockState, int] = {}
        states_by_photon_count: dict[int, list[int]] = {}
        for i, comp in enumerate(sub_comps):
            state = comp.experiment.input_state
            states_by_photon_count.setdefault(state.n, []).append(i)
            state_idx[state] = i

        input_state = computation.experiment.input_state
        order = self._resolve_order(input_state.n)

        pnr_per_mode = [d.max_detections if d is not None else None for d in imperfections.detectors][:input_state.m]
        detection_type = get_detection_type(imperfections.detectors[:input_state.m])

        dist_batch = self._extract_distributions(results)
        mitigated = self._mitigate_hom(
            dist_batch,
            order,
            state_idx,
            input_state,
            noise.indistinguishability,
            pnr_per_mode,
            detection_type
        )
        dist_batch[0] = mitigated

        mitigated = self._mitigate_g2(
            dist_batch,
            order,
            states_by_photon_count,
            input_state,
            noise,
            pnr_per_mode,
            detection_type
        )
        mitigated.normalize()

        parsed = copy(results[0])
        parsed[KEY_RESULTS] = mitigated

        return parsed

    @staticmethod
    def _copy_computation(
        computation: Computation,
        input_state: FockState,
        samples: int | None,
        shots: int | None,
        job_name: str
    ) -> Computation:
        """Copy a computation with a new input state & samples/shots.
        """
        comp = deepcopy(computation)
        comp.job_name = job_name
        if samples is not None:
            comp.add_params(**{KEY_MAX_SAMPLES: samples})
        if shots is not None:
            comp.add_params(**{KEY_MAX_SHOTS: shots})

        comp.command.name = "probs"

        experiment = comp.experiment
        experiment.remove_all_ports()  # Remove heralds
        experiment.clear_postselection()

        total_n = experiment.input_state.n
        experiment.with_input(input_state)

        # Filtering like this will remove all states such that the products will give less than the user's min_photons
        min_filter = experiment.min_photons_filter or 0
        experiment.min_detected_photons_filter(max(min_filter - (total_n - input_state.n), 1))
        # TODO: estimate the right splitting of samples given this definition

        return comp

    @staticmethod
    def _validate_input_state(input_state) -> FockState:
        if input_state is None:
            raise ValueError("PhotonErrorMitigation requires the experiment to have an input state.")

        if not isinstance(input_state, FockState):
            raise TypeError(f"PhotonErrorMitigation requires a fixed FockState input (got {type(input_state).__name__}).")

        return input_state

    @staticmethod
    def _validate_order(order: int | dict[int, int]):
        if isinstance(order, int):
            if order <= 0:
                raise ValueError("order must be an integer greater than 0.")

            if order > 4:
                get_logger().warn(
                    "High order may result in incomplete jobs on the cloud due to time-out. Use with caution.",
                )

        elif isinstance(order, dict):
            warned = False
            for k, v in order.items():
                if not isinstance(k, int):
                    raise ValueError("order keys must be integers.")

                if k < 0:
                    raise ValueError("order keys must be non-negative.")

                if not isinstance(v, int):
                    raise ValueError("order values must be integers.")

                if v <= 0:
                    raise ValueError("order must be an integer greater than 0.")

                if v > 4 and not warned:
                    warned = True
                    get_logger().warn(
                        "High order may result in incomplete jobs on the cloud due to time-out. Use with caution.",
                    )

        else:
            raise TypeError("Wrong type received for 'order'. "
                            f"Received {type(order).__name__}., expected int or dict[int, int]")


    def _resolve_order(self, photon_count: int) -> int:
        """Resolve int/dict order and clamp it to the photon count.

        Order `n-1` is promoted to `n` because it has no extra cost.
        """
        if isinstance(self._order, dict):
            if photon_count not in self._order:
                raise ValueError("Given photon count is not known to the wanted order.")
            order = self._order[photon_count]
        else:
            order = self._order

        return photon_count if order >= photon_count - 1 else order

    @staticmethod
    def _extract_distributions(results: list[dict]):
        res = []
        for sub_res in results:
            dist = sub_res["results"]

            # Now estimates the "unwanted states" probability and adds it as 0-photon state.
            # None of these states would have contributed to anything useful,
            # except the 0-photon state if min_photons is 1, in which case this is the only state represented by the global perf
            dist = sub_res["global_perf"] * dist + (1 - sub_res["global_perf"]) * BSDistribution(FockState(dist.m))
            res.append(dist)
        return res

    @classmethod
    def _mitigate_hom(
        cls,
        dist_batch: list[BSDistribution],
        order: int,
        state_idx: dict[FockState, int],
        input_state: FockState,
        indistinguishability: float,
        pnr_per_mode: list[int | None],
        detection_type: DetectionType,
    ) -> BSDistribution:
        """Mitigate distinguishability by subtracting contributions where
        photons statistics are independent.
        """
        if indistinguishability == 1:
            return dist_batch[state_idx[input_state]]

        photon_count = input_state.n
        order = min(photon_count, order)
        weights_hom = cls._compute_weights_hom(indistinguishability, photon_count, order)

        res = weights_hom[0] * dist_batch[state_idx[input_state]]
        for i in range(1, order + 1):
            # TODO: avoid tensor product for order n (cell is identical to order n-1)
            for cell, multiplicity in generate_obb_partition(input_state, i):
                convolved = BSDistribution.list_tensor_product(
                    [dist_batch[state_idx[state]] for state in cell],
                    merge_modes=True
                )
                # It could be more efficient to apply this once at the end, both in terms of speed and correctness
                # This would be at the cost of memory
                convolved = apply_detection_filter(convolved, pnr_per_mode, detection_type)
                res += weights_hom[i] * multiplicity * convolved

        return res

    @classmethod
    def _mitigate_g2(
        cls,
        dist_batch: list[BSDistribution],
        order: int,
        idx_by_photon_count: dict[int, list[int]],
        input_state: FockState,
        noise: NoiseModel,
        pnr_per_mode: list[int | None],
        detection_type: DetectionType
    ) -> BSDistribution:
        """Mitigate g2 in computation by subtracting statistics due to extra
        distinguishable photon in lossy subspace.
        """
        if noise.g2 == 0:
            return dist_batch[idx_by_photon_count[input_state.n][0]]

        photon_count = input_state.n
        order = min(order, max(photon_count - 1, 0))
        weights_g2 = cls._compute_weights_g2(noise, photon_count, order)

        res = weights_g2[0] * dist_batch[idx_by_photon_count[input_state.n][0]]
        for i in range(1, order + 1):
            signal_dists = [
                dist_batch[idx]
                for idx in idx_by_photon_count.get(photon_count - i, [])
            ]
            noise_dists = [
                dist_batch[idx]
                for idx in idx_by_photon_count.get(i, [])
            ]

            convolved = BSDistribution.tensor_product(
                sum(signal_dists, BSDistribution()),
                sum(noise_dists, BSDistribution()),
                merge_modes=True,
            )
            # It could be more efficient to apply this once at the end, both in terms of speed and correctness
            # This would be at the cost of memory
            convolved = apply_detection_filter(convolved, pnr_per_mode, detection_type)
            res += weights_g2[i] * convolved

        # Filter out g2 states. In theory, there shouldn't be any left, but it's better to be sure about that
        return filter_extra_photons(res, input_state.n)

    @staticmethod
    def _compute_weights_hom(
        indistinguishability: float,
        photon_count: int,
        order: int
    ) -> list[float]:
        if photon_count == 0:
            return [1]

        g = math.sqrt(indistinguishability)
        b = 1 - g

        return [(-1) ** i * b ** i for i in range(min(photon_count, order) + 1)]

    @staticmethod
    def _compute_weights_g2(
        noise: NoiseModel | None,
        photon_count: int,
        order: int
    ) -> list[float]:
        noise = noise or NoiseModel()
        g2 = noise.g2

        if g2:
            p2 = (1 - math.sqrt(1 - 2 * g2) - g2) / g2
        else:
            p2 = 0

        # The loss is not exactly this product, but this is accurate enough for our needs
        loss = 1 - noise.transmittance * noise.brightness
        return [1] + [
            -(p2 * loss) ** i * (1 - p2) ** (photon_count - i)
            for i in range(1, order + 1)
        ]

    @staticmethod
    def _split_ratios(states: list[FockState], transmittance: float):
        if transmittance <= 0:
            raise ValueError("Can't do anything with 0 transmittance.")

        norm = sum(transmittance ** (-state.n / 2) for state in states)
        return [transmittance ** (-state.n / 2) / norm for state in states]

    @staticmethod
    def _split_integer(total: int | None, ratios: list[float], name: str):
        """Split an integer `total` into amounts determined by `ratios`.
        """
        if total is None:
            return [None] * len(ratios)

        if total < len(ratios):
            raise RuntimeError(
                f"PhotonErrorMitigation: cannot split {name}={total} over "
                f"{len(ratios)} sub-computations"
            )

        total_ratio = sum(ratios)
        assert total_ratio > 0, "Split ratios are negative."

        ratios = [ratio / total_ratio for ratio in ratios]

        # Give every sub-computation one unit, then apportion the rest.
        values = [1] * len(ratios)
        remaining = total - len(ratios)

        raw_extra = [remaining * ratio for ratio in ratios]
        extra = [math.floor(value) for value in raw_extra]
        values = [
            value + extra_value
            for value, extra_value in zip(values, extra)
        ]

        # Assign leftover units to the largest fractional remainders.
        leftover = total - sum(values)
        fractions = [
            (raw_extra[i] - math.floor(raw_extra[i]), i)
            for i in range(len(ratios))
        ]
        for _, i in sorted(fractions, reverse=True)[:leftover]:
            values[i] += 1

        return values


Serialization.register_class(DistinguishablePhotonMitigation, ["_order"])
