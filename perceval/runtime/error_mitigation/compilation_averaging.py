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

from .abstract_mitigation import AbstractMitigation
from ..computation import Computation

from perceval.utils import NoiseModel, BSCount
from perceval.utils.constants import KEY_MAX_SHOTS, KEY_MAX_SAMPLES, KEY_SHOTS_USED, KEY_GLOBAL_PERF, KEY_PHYSICAL_PERF, \
    KEY_LOGICAL_PERF, KEY_RESULTS

class CompilationAveraging(AbstractMitigation):

    APPLY_MIN_PHOTONS = False
    APPLY_LOGICAL_SELECTION = False

    def __init__(self, repetitions: int, starting_seed: int = None):
        """
        A mitigation process that splits the requested computation into :code:`repetitions` sub-computations,
        where the requested shots and samples are equally divided, asking for a new compilation seed every time.

        At post-processing, it adds up the results.

        :param repetitions: The number of subdivisions. The greater this number, the more time will be spent on compilation
        :param starting_seed: Optional, seed to use as a starting point for the compilation seed.
        """
        self.repetitions = repetitions
        assert isinstance(self.repetitions, int) and repetitions >= 1,\
            f"Number of repetitions must be a positive integer (got {repetitions})"
        self.starting_seed = starting_seed

    def extend_computation(self, computation: Computation, noise: NoiseModel) -> list[Computation]:
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
            new_comp.command.name = "sample_count"

            if shots_per_computation is not None:
                new_comp.add_params(max_shots=shots_per_computation + (i < remaining_shots))
            if samples_per_computation is not None:
                new_comp.add_params(max_samples=samples_per_computation + (i < remaining_samples))
            new_comp.add_params(compilation_seed=starting_seed + i)

            res.append(new_comp)

        return res

    def _parse_results(self, computation: Computation, results: list[dict], noise: NoiseModel) -> dict:
        # First, do nothing if nothing was done - for example no compilation seed could be set
        if len(results) == 1:
            return results[0]

        # Here, we know we have expanded the computation, so all results are BSCount
        bsc = BSCount()

        # global_perf = n_samples / n_clock; phys_perf = n_phys / n_clock; log_perf = n_samples / n_phys
        n_clocks = 0
        n_physical = 0
        shots_used = 0

        for res in results:
            res_bsc: BSCount = res[KEY_RESULTS]
            for state, count in res_bsc.items():
                bsc[state] += count

            if shots_used is not None and KEY_SHOTS_USED in res:
                shots_used += res[KEY_SHOTS_USED]
            else:
                shots_used = None

            sub_n_clocks = res_bsc.total() / res[KEY_GLOBAL_PERF]
            n_clocks += sub_n_clocks

            if n_physical is not None and KEY_PHYSICAL_PERF in res:
                n_physical += sub_n_clocks * res[KEY_PHYSICAL_PERF]
            else:
                n_physical = None

        res = copy(results[0])  # We are going to modify this to keep custom fields as much as we can
        res[KEY_RESULTS] = bsc
        n_samples = bsc.total()
        res[KEY_GLOBAL_PERF] = n_samples / n_clocks

        if n_physical is not None:
            res[KEY_PHYSICAL_PERF] = n_physical / n_clocks
            res[KEY_LOGICAL_PERF] = n_samples / n_physical

        if shots_used is not None:
            res[KEY_SHOTS_USED] = shots_used

        return res
