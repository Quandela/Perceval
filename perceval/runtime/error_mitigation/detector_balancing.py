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

from copy import deepcopy, copy

from perceval.utils import BSDistribution
from perceval.simulators._simulate_detectors import compute_distributions

from .abstract_mitigation import AbstractMitigation
from .imperfections import Imperfections
from ..computation import Computation

from perceval.utils.constants import KEY_RESULTS, KEY_GLOBAL_PERF, KEY_PHYSICAL_PERF
from perceval.serialization import Serialization

class DetectorBalancing(AbstractMitigation):
    """
    A mitigation process that adjusts the probabilities of each output state based on the output
    loss and number of photons in each mode.
    """
    # Hypotheses for this mitigation to work:
    #   - the detectors only model photon losses (no dark count)
    #   - the detectors act independently for each state (no dependency between states)
    # They are verified by all the detectors models in perceval version 1.3
    # The closer a detector's model is to it's true behavior, the better the correction will be
    # One last hypothesis:
    #   - all states are represented in the results (i.e. bunched states are also represented)
    # Not fulfilling this hypothesis will make the maths wrong, but it should still be better than without mitigation


    APPLY_MIN_PHOTONS = False
    APPLY_LOGICAL_SELECTION = True  # Actually, all we need is "at least", but it's easier to remove everything

    def extend_computation(self, computation: Computation, imperfections: Imperfections) -> list[Computation]:
        detectors = imperfections.detectors
        assert len(detectors) >= computation.experiment.circuit_size

        comp = deepcopy(computation)
        comp.command.name = "probs"

        comp.experiment.remove_all_ports()
        comp.experiment.clear_postselection()
        return [comp]

    def _parse_results(self, computation: Computation, results: list[dict], imperfections: Imperfections) -> dict:
        detectors = imperfections.detectors[:computation.experiment.circuit_size]

        res_by_n = []
        for state, prob in results[0][KEY_RESULTS].items():
            for _ in range(state.n + 1 - len(res_by_n)):
                res_by_n.append({})
            res_by_n[state.n][state] = prob

        final_res = BSDistribution()
        perf_factor = 0   # May be used by DistinguishablePhotonMitigation to get the 0-photon probability
        for n in range(len(res_by_n) - 1, -1, -1):
            for state in res_by_n[n].keys():
                distributions = compute_distributions(state, detectors, {})

                # prob threshold ?
                state_dist = BSDistribution.list_tensor_product(distributions)

                # If we were able to detect state, it means that the detectors model should have state in its results
                if state not in state_dist:
                    raise RuntimeError(f"Measured state {state} can't be obtained through the detectors model. "
                                       "Are the detectors the ones that were used to obtain this distribution?")

                # At this stage, p_measured = p_theoretical * p_detection
                res_by_n[n][state] /= state_dist[state]

                state_prob = res_by_n[n][state]
                for sub_state, sub_prob in state_dist.items():
                    # For sub states, p_measured(sub_state) = ... + p_theoretical(state) * p_detection(sub_state | state)
                    if sub_state != state and sub_state in res_by_n[sub_state.n]:
                        res_by_n[sub_state.n][sub_state] -= sub_prob * state_prob
                        if res_by_n[sub_state.n][sub_state] <= 0:
                            del res_by_n[sub_state.n][sub_state]

                final_res[state] = state_prob
                perf_factor += state_prob

        final_res.normalize()

        res = copy(results[0])  # We are going to modify this to keep custom fields as much as we can
        res[KEY_RESULTS] = final_res
        res[KEY_GLOBAL_PERF] *= perf_factor  # May makes the perf bigger than 1, but should theoretically not do it
        if KEY_PHYSICAL_PERF in res:
            res[KEY_PHYSICAL_PERF] *= perf_factor

        return res


Serialization.register_class(DetectorBalancing, [])
