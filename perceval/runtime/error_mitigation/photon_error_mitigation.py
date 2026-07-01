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
import warnings
from copy import copy, deepcopy
from typing import Callable

from perceval.utils import (
    BSDistribution,
    ConversionHelper,
    FockState,
    NoiseModel,
)
from perceval.utils.constants import (
    KEY_GLOBAL_PERF,
    KEY_MAX_SAMPLES,
    KEY_MAX_SHOTS,
    KEY_LOGICAL_PERF,
    KEY_PHYSICAL_PERF,
    KEY_RESULTS,
    KEY_SHOTS_USED,
)

from ..computation import Computation
from .abstract_mitigation import AbstractMitigation
from .utils._photon_error_mitigation import (
    _generate_obb_partition, _generate_obb_states,
    _apply_detection_filter,
    _extract_photon_number
)


class PhotonErrorMitigation(AbstractMitigation):
    """
    Partial distinguishability and g2 mitigation.

    Mitigates distinguishability and g2 errors by preparing jobs with fewer
    photons and recombining them through corrections based on the partial
    distinguishability 'orthogonal bad bits' model.

    :param order: Extent of photon error mitigation. If an integer is given,
        the correction is fixed up to ``order`` or the input photon number.
        If a callable is given, it is evaluated on the input photon number.
    """

    def __init__(self, order: int | Callable[[int], int]):
        self._validate_order(order)
        self._order = order
        self._settings = {}

    def overhead(self, input_state: FockState) -> int:
        """Return the number of sub-computations needed for a given input
        state based on the instance order.

        :param input_state: Provided input-state or number of photons.
        """
        input_state = self._validate_input_state(input_state)

        order = self._resolve_order(input_state.n)
        return len(_generate_obb_states(input_state, order))

    def extend_computation(
        self,
        computation: Computation,
        noise: NoiseModel
    ) -> list[Computation]:
        input_state = self._validate_input_state(
            computation.experiment.input_state
        )

        resolved_order = self._resolve_order(input_state.n)
        new_input_states = _generate_obb_states(input_state, resolved_order)
        mitigation_noise = computation.experiment.noise or noise
        ratios = self._split_ratios(new_input_states, mitigation_noise)

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

        state_idx = {}
        sub_parameters = []

        sub_computations = []
        for i, state in enumerate(new_input_states):
            state_idx[state] = i

            comp = self._copy_computation(
                computation,
                input_state=state,
                samples=samples[i],
                shots=shots[i],
                job_name=f"{computation.job_name} pem {i + 1}",
            )
            sub_parameters.append(copy(comp.parameters))
            sub_computations.append(comp)

        # Save some settings for _parse_results
        self._settings[computation] = (
            input_state,
            resolved_order,
            state_idx,
            sub_parameters
        )

        return sub_computations

    def _parse_results(
        self,
        computation: Computation,
        results: list[dict],
        noise: NoiseModel
    ) -> dict:
        input_state, order, state_idx, sub_parameters = self._settings[computation]

        if len(results) == 1 and state_idx is None:
            return results[0]

        mitigation_noise = computation.experiment.noise or noise

        # TODO: Access PNR per mode in a way that supports local & QPU computation
        pnr_per_mode = ...

        dist_batch = self._prepare_distribution_batch(
            results,
            sub_parameters,
            state_idx,
        )

        mitigated = self._mitigate_hom(
            dist_batch,
            order,
            state_idx,
            input_state,
            mitigation_noise,
            pnr_per_mode,
        )
        dist_batch[0] = mitigated

        mitigated = self._mitigate_g2(
            dist_batch,
            order,
            state_idx,
            input_state,
            mitigation_noise,
            pnr_per_mode,
        )
        mitigated.normalize()

        parsed = copy(results[0])
        parsed[KEY_RESULTS] = mitigated

        n_samples = computation.parameters.get(KEY_MAX_SAMPLES)
        max_shots = computation.parameters.get(KEY_MAX_SHOTS)
        if n_samples is None or (
                max_shots is not None and max_shots < n_samples):
            n_samples = max_shots

        n_clocks = 0
        n_physical = 0
        shots_used = 0
        for result, parameters in zip(results, sub_parameters):
            sub_samples = parameters.get(KEY_MAX_SAMPLES)
            sub_shots = parameters.get(KEY_MAX_SHOTS)
            if sub_samples is None or (
                    sub_shots is not None and sub_shots < sub_samples):
                sub_samples = sub_shots

            if sub_samples is None:
                n_clocks = None
            if n_clocks is not None:
                sub_n_clocks = sub_samples / result[KEY_GLOBAL_PERF]
                n_clocks += sub_n_clocks

                if n_physical is not None and KEY_PHYSICAL_PERF in result:
                    n_physical += sub_n_clocks * result[KEY_PHYSICAL_PERF]
                else:
                    n_physical = None

            if shots_used is not None and KEY_SHOTS_USED in result:
                shots_used += result[KEY_SHOTS_USED]
            else:
                shots_used = None

        if n_samples is not None and n_clocks:
            parsed[KEY_GLOBAL_PERF] = n_samples / n_clocks

            if n_physical is not None:
                parsed[KEY_PHYSICAL_PERF] = n_physical / n_clocks
                parsed[KEY_LOGICAL_PERF] = n_samples / n_physical

        if shots_used is not None:
            parsed[KEY_SHOTS_USED] = shots_used
        return parsed

    def _copy_computation(
        self,
        computation: Computation,
        input_state: FockState,
        samples: int | None,
        shots: int | None,
        job_name: str
    ) -> Computation:
        comp = deepcopy(computation)
        comp.job_name = job_name
        if samples is not None:
            comp.add_params(**{KEY_MAX_SAMPLES: samples})
        if shots is not None:
            comp.add_params(**{KEY_MAX_SHOTS: shots})

        experiment = comp.experiment
        experiment._input_state = input_state
        experiment._input_changed()

        return comp

    @staticmethod
    def _prepare_distribution_batch(
        results: list[dict],
        sub_parameters: list[dict],
        state_idx: dict[FockState, int],
    ) -> list[BSDistribution]:
        dist_batch = [BSDistribution() for _ in results]
        for state, idx in state_idx.items():
            converted = ConversionHelper.convert_to(
                "probs",
                results[idx][KEY_RESULTS],
                **sub_parameters[idx],
            )
            dist = BSDistribution(dict(converted.items()))
            if len(dist):
                dist.normalize()
            dist_batch[idx] = _extract_photon_number(dist, state.n)

        return dist_batch

    @staticmethod
    def _validate_input_state(input_state) -> FockState:
        if input_state is None:
            raise ValueError(
                "PhotonErrorMitigation requires the experiment to have an "
                "input state."
            )

        if not isinstance(input_state, FockState):
            raise TypeError(
                "PhotonErrorMitigation requires a fixed FockState input "
                f"(got {type(input_state).__name__})."
            )

        return input_state

    @staticmethod
    def _validate_order(order):
        if callable(order):
            return

        if not isinstance(order, int) or order <= 0:
            raise ValueError("order must be an integer greater than 0.")

        if order > 4:
            warnings.warn(
                "High order may result in incomplete jobs on the cloud "
                "due to time-out. Use with caution.",
                UserWarning,
                stacklevel=2,
            )

    def _resolve_order(self, photon_count: int) -> int:
        """Resolve int/callable order and clamp it to the photon count.

        Order ``n - 1`` is promoted to ``n`` because it has no extra cost.
        """
        order = (
            self._order(photon_count)
            if callable(self._order)
            else self._order
        )
        if not isinstance(order, int) or order <= 0:
            raise ValueError("order must be an integer greater than 0.")

        return photon_count if order >= photon_count - 1 else order

    @classmethod
    def _mitigate_hom(
        cls,
        dist_batch: list[BSDistribution],
        order: int,
        state_idx: dict[FockState, int],
        input_state: FockState,
        noise: NoiseModel,
        pnr_per_mode: list[int]
    ) -> BSDistribution:
        """Mitigate distinguishability in a single results with respect to
        its helper batch.
        """
        photon_count = input_state.n
        order = min(photon_count, order)
        corrections = [dist_batch[0]]
        weights_hom = cls._compute_weights_hom(noise, photon_count, order)
        partition_counts = {0: 1}

        for i in range(1, order + 1):
            partitions = _generate_obb_partition(input_state, i)
            partition_counts[i] = len(partitions)

            for cell in partitions:
                convolved = BSDistribution.list_tensor_product(
                    [dist_batch[state_idx[state]] for state in cell],
                    merge_modes=True
                )
                convolved = _apply_detection_filter(convolved, pnr_per_mode)
                convolved = _extract_photon_number(convolved, photon_count)
                corrections.append(convolved)

        weights_hom = [
            weight
            for i, weight in enumerate(weights_hom)
            for _ in range(partition_counts[i])
        ]

        if len(corrections) != len(weights_hom):
            raise RuntimeError(
                "Number of indistinguishability corrections does not "
                "match number of weights."
            )

        return sum(
            (weight * dist for dist, weight in zip(corrections, weights_hom)),
            BSDistribution(),
        )

    @classmethod
    def _mitigate_g2(
        cls,
        dist_batch: list[BSDistribution],
        order: int,
        state_idx: dict[FockState, int],
        input_state: FockState,
        noise: NoiseModel,
        pnr_per_mode: list[int]
    ) -> BSDistribution:
        photon_count = input_state.n
        order = min(order, max(photon_count - 1, 0))
        weights_g2 = cls._compute_weights_g2(noise, photon_count, order)
        corrections = [dist_batch[0]]

        for i in range(1, order + 1):
            signal_dists = []
            noise_dists = []
            for state, idx in state_idx.items():
                if state.n == i:
                    noise_dists.append(dist_batch[idx])
                if state.n == photon_count - i:
                    signal_dists.append(dist_batch[idx])

            # 1 * dist makes a copy to avoid in place mutation
            convolved = BSDistribution.list_tensor_product(
                [sum((1 * dist for dist in signal_dists), BSDistribution()),
                 sum((1 * dist for dist in noise_dists), BSDistribution())],
                merge_modes=True)
            convolved = _apply_detection_filter(convolved, pnr_per_mode)
            convolved = _extract_photon_number(convolved, photon_count)
            corrections.append(convolved)

        return sum((
                weight * dist
                for dist, weight in zip(corrections, weights_g2)),
            BSDistribution(),
        )

    @staticmethod
    def _compute_weights_hom(
        noise: NoiseModel | None,
        photon_count: int,
        order: int
    ) -> list[float]:
        noise = noise or NoiseModel()
        if photon_count == 0:
            return [1]

        g = math.sqrt(noise.indistinguishability)
        b = 1 - g

        weights = [(-1) ** i * b ** i for i in range(photon_count - 1)]
        weights.append((-1) ** (1 + photon_count % 2)
                       * b ** (photon_count - 1))
        weights.append((-1) ** (photon_count % 2) * b ** photon_count)
        return weights[:order + 1]

    @staticmethod
    def _compute_weights_g2(
        noise: NoiseModel | None,
        photon_count: int,
        order: int
    ) -> list[float]:
        noise = noise or NoiseModel()
        g2 = noise.g2

        if g2 > .5:
            raise ValueError("PhotonErrorMitigation requires g2 <= 0.5.")

        if g2:
            p2 = (1 - math.sqrt(1 - 2 * g2) - g2) / g2
        else:
            p2 = 0

        loss = 1 - noise.transmittance
        return [1] + [
            -(p2 * loss) ** i * (1 - p2) ** (photon_count - i)
            for i in range(1, order + 1)
        ]

    @staticmethod
    def _split_ratios(states: list[FockState], noise: NoiseModel | None):
        noise = noise or NoiseModel()
        transmittance = noise.transmittance
        assert transmittance > 0, "Improper calibration has led to zero transmittance."

        norm = sum(transmittance ** (-state.n / 2) for state in states)
        return [transmittance ** (-state.n / 2) / norm for state in states]

    @staticmethod
    def _split_integer(total: int | None, ratios: list[float], name: str):
        if total is None:
            return [None] * len(ratios)

        if total < len(ratios):
            raise RuntimeError(
                f"PhotonErrorMitigation: cannot split {name}={total} over "
                f"{len(ratios)} sub-computations"
            )

        assert sum(ratios) > 0, "Split ratios are negative."

        ratios = [ratio / sum(ratios) for ratio in ratios]

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
