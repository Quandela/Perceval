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
from copy import deepcopy

import numpy as np
from math import comb

from scipy.optimize import curve_fit

from perceval.components import PortLocation
from perceval.utils import BSCount, BSDistribution, BasicState, SVDistribution, NoiseModel, PostSelect
from perceval.utils.logging import get_logger, channel

from ..computation import Computation
from .abstract_mitigation import AbstractMitigation
from ._loss_mitigation_utils import _gen_lossy_dists, _get_avg_exp_from_uni_dist, _generate_one_photon_per_mode_mapping


def _validate_noisy_input(noisy_input: BSCount | BSDistribution, ideal_photon_count: int):
    if not isinstance(noisy_input, (BSCount, BSDistribution)):
        # check if the input type is correct
        raise TypeError('Noisy input should be of type BSCount or BSDistribution')

    if all([states.n == ideal_photon_count for states in noisy_input.keys()]):
        # Check if a perfect loss-less distribution was passed to mitigate
        raise ValueError("All input states have ideal photon count, no loss to mitigate.")

    target_photon_loss = {ideal_photon_count-1, ideal_photon_count-2}
    obtained_photon_counts = {states.n for states in noisy_input.keys()}
    if not target_photon_loss.issubset(obtained_photon_counts):
        # check if noisy input meets current implementation's statistics criteria
        raise ValueError("Noisy input invalid or incorrect ideal photon count, "
                         "implementation requires states with n-1, n-2 photons for expected n-photon mitigation")


def photon_recycling(noisy_input: BSCount | BSDistribution, ideal_photon_count: int) -> BSDistribution:
    """
    A classical technique to mitigate errors in the output distribution caused by photon
    loss in LO quantum circuits (ref: https://arxiv.org/abs/2405.02278)

    :param noisy_input: Noisy output (Basic State Samples or a distribution)
    :param ideal_photon_count: expected photon count for a loss-less system
    :return: photon loss mitigated distribution
    """
    get_logger().info(f"Running Photon Recycling on a {len(noisy_input)} states distribution targeting {ideal_photon_count} ideal photons",
                channel.general)
    # run checks on noisy input before recycling
    _validate_noisy_input(noisy_input, ideal_photon_count)

    m = next(iter(noisy_input)).m  # number of modes
    pattern_map = _generate_one_photon_per_mode_mapping(m, ideal_photon_count)
    noisy_distributions = _gen_lossy_dists(noisy_input, ideal_photon_count, pattern_map)

    # GET AVERAGE EXPONENT USING AVERAGE DISTANCE FROM UNIFORM PROBABILITY
    z = _get_avg_exp_from_uni_dist(noisy_distributions, m, ideal_photon_count)
    median_of_means = z[0]

    # Generating the mitigated distribution using the decay parameter.
    mitigated_probs = []
    c_mn_inv = 1 / comb(m, ideal_photon_count)

    def func1(x, a):
        return a * np.exp(-median_of_means * x) + c_mn_inv

    for k in range(len(noisy_distributions[0])):
        z, _ = curve_fit(f=func1,
                         xdata=[1, 2, 50],
                         ydata=[noisy_distributions[1][k], noisy_distributions[2][k], c_mn_inv],
                         bounds=([-5], [5]),
                         maxfev=2000000)

        if noisy_distributions[1][k] > c_mn_inv > noisy_distributions[2][k]:
            mitigated_probs.append(c_mn_inv)
        elif noisy_distributions[1][k] < c_mn_inv < noisy_distributions[2][k]:
            mitigated_probs.append(c_mn_inv)
        else:
            mitigated_probs.append(func1(0, z[0]))

    mitigated_probs = [0 if i < 0 else i for i in mitigated_probs]
    mitigated_probs = mitigated_probs / np.sum(mitigated_probs)

    mitigated_distribution = BSDistribution()
    for index, keys in enumerate(pattern_map.keys()):
        state = BasicState(keys)
        mitigated_distribution.add(state, mitigated_probs[index])

    return mitigated_distribution


class PhotonRecycling(AbstractMitigation):
    """
    A mitigation layer that performs the photon recycling algorithm internally
    """

    @staticmethod
    def _ideal_photon_number(computation: Computation) -> int:
        # Returns 0 if the experiment is not compatible
        exp = computation.experiment

        if not exp.is_unitary:   # I have to admit I don't know if the algorithm can apply in this case
            return 0

        if isinstance(exp.input_state, BasicState):
            return exp.input_state.n

        if isinstance(exp.input_state, SVDistribution):  # Nor here
            all_n = set()
            for state in exp.input_state:
                all_n |= state.n

            if len(all_n) == 1:
                return all_n.pop()

        return 0

    def extend_computation(self, computation: Computation, noise: NoiseModel) -> list[Computation]:
        expected_photons = self._ideal_photon_number(computation)

        if expected_photons < 3:
            get_logger().info(f"Photon recycling disabled: not enough photons to mitigate ({expected_photons} < 3), "
                              "or the given Experiment is not compatible")
            return [computation]

        comp = deepcopy(computation)
        if comp.experiment.post_select_fn.has_condition or len(comp.experiment.heralds):
            get_logger().warn("Photon recycling is used along logical post-selection. Not optimal.", channel.user)
            comp.experiment.set_postselection(PostSelect())
            for m in comp.experiment.heralds:
                comp.experiment.remove_port(m, location=PortLocation.OUTPUT)

        if comp.experiment.min_photons_filter is None or comp.experiment.min_photons_filter > expected_photons - 2:
            comp.experiment.min_detected_photons_filter(expected_photons - 2)

        comp.command.name = "probs"

        return [comp]

    def _parse_results(self, computation: Computation, results: list[dict], noise: NoiseModel) -> dict:
        ideal_photon_count = self._ideal_photon_number(computation)
        res = results[0]

        if ideal_photon_count < 3:
            return res

        try:
            res["results"] = photon_recycling(res["results"], ideal_photon_count)
        except ValueError:
            pass

        return res
