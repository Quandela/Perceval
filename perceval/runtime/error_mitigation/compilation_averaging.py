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

import random
from copy import deepcopy, copy

from .abstract_mitigation import AMitigation
from .imperfections import Imperfections
from ..computation import Computation

from perceval.utils import BSDistribution
from perceval.utils.constants import KEY_MAX_SHOTS, KEY_MAX_SAMPLES, KEY_GLOBAL_PERF, KEY_PHYSICAL_PERF, \
    KEY_LOGICAL_PERF, KEY_RESULTS
from perceval.serialization import Serialization

class CompilationAveraging(AMitigation, tag="CompilationAveraging"):
    """Reduce sensitivity to a single physical compilation by averaging the results over several compilations.

    The requested samples and shots are divided between ``repetitions`` sub-computations, each
    using a different compilation seed. Their sample counts are combined during post-processing.
    Commands that do not accept a ``compilation_seed`` parameter are left unchanged.

    :param repetitions: Number of compilations to average. More repetitions require more compilation work.
    :param starting_seed: Optional first compilation seed. When omitted, choose one randomly.
    """

    APPLY_MIN_PHOTONS = False
    APPLY_LOGICAL_SELECTION = False

    def __init__(self, repetitions: int, starting_seed: int = None):
        self.repetitions = repetitions
        assert isinstance(self.repetitions, int) and repetitions >= 1, \
            f"Number of repetitions must be a positive integer (got {repetitions})"
        self.starting_seed = starting_seed

    def extend_computation(self, computation: Computation, imperfections: Imperfections) -> list[Computation]:
        if not any(signature[0] == "compilation_seed" for signature in computation.command.signature):
            return [computation]  # Can't do anything

        starting_seed = self.starting_seed if self.starting_seed is not None else random.randint(0, 1_000_000)

        shots: int | None = computation.parameters.get(KEY_MAX_SHOTS)
        if shots is not None:
            if shots < self.repetitions:
                raise RuntimeError("CompilationAveraging: Can't split into more sub-computations than the number of shots")
            shots_per_computation = shots // self.repetitions
            remaining_shots = shots - shots_per_computation * self.repetitions
        else:
            shots_per_computation = None
            remaining_shots = None

        samples: int | None = computation.parameters.get(KEY_MAX_SAMPLES)
        if samples is not None:
            if samples < self.repetitions:
                raise RuntimeError("CompilationAveraging: Can't split into more sub-computations than the number of samples")
            samples_per_computation = samples // self.repetitions
            remaining_samples = samples - samples_per_computation * self.repetitions
        else:
            samples_per_computation = None
            remaining_samples = None

        res = []
        for i in range(self.repetitions):
            new_comp = deepcopy(computation)
            new_comp.command.name = "probs"

            if shots_per_computation is not None:
                new_comp.add_params(max_shots=shots_per_computation + (i < remaining_shots))
            if samples_per_computation is not None:
                new_comp.add_params(max_samples=samples_per_computation + (i < remaining_samples))
            new_comp.add_params(compilation_seed=starting_seed + i)

            res.append(new_comp)

        return res

    def _parse_results(self, computation: Computation, results: list[dict], imperfections: Imperfections) -> dict:
        # First, do nothing if nothing was done - for example no compilation seed could be set
        if len(results) == 1:
            return results[0]

        # Here, we know we have expanded the computation, so all results are BSDistributions
        bsd = BSDistribution()
        global_perf = 0
        physical_perf = 0
        logical_perf = 0

        for res in results:
            bsd += res[KEY_RESULTS]
            global_perf += res[KEY_GLOBAL_PERF]

            if physical_perf is not None and KEY_PHYSICAL_PERF in res:
                physical_perf += res[KEY_PHYSICAL_PERF]
            else:
                physical_perf = None

            if logical_perf is not None and KEY_LOGICAL_PERF in res:
                logical_perf += res[KEY_LOGICAL_PERF]
            else:
                logical_perf = None

        res = copy(results[0])  # We are going to modify this to keep custom fields as much as we can

        bsd.normalize()
        res[KEY_RESULTS] = bsd
        res[KEY_GLOBAL_PERF] = global_perf / len(results)

        # Note: we lose the fact that phys_perf * log_perf = global_perf
        if physical_perf is not None:
            res[KEY_PHYSICAL_PERF] = physical_perf / len(results)
        if logical_perf is not None:
            res[KEY_LOGICAL_PERF] = logical_perf / len(results)

        return res


Serialization.register_class(CompilationAveraging, ["repetitions", "starting_seed"])
