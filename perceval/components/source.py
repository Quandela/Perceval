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
from __future__ import annotations

import math

from perceval.utils import (SVDistribution, StateVector, BasicState, anonymize_annotations, NoiseModel, global_params,
                            BSDistribution)
from perceval.utils.logging import get_logger, channel

DISTINGUISHABLE_KEY = 'distinguishable'
INDISTINGUISHABLE_KEY = 'indistinguishable'

class Source:
    r"""Definition of a source
    We build on a phenomenological model first introduced in ref. [1] where an imperfect quantum-dot based single-photon
    source is modeled by a statistical mixture of Fock states. The model developed here, first introduced in ref. [2],
    constructs the input multi-photon state using features specific to Perceval.
    [1] Pont, Mathias, et al. Physical Review X 12, 031033 (2022). https://doi.org/10.1103/PhysRevX.12.031033
    [2] Pont, Mathias, et al. arXiv preprint arXiv:2211.15626 (2022). https://doi.org/10.48550/arXiv.2211.15626

    :param emission_probability: probability that the source emits at least one photon
    :param multiphoton_component: second order intensity autocorrelation at zero time delay :math:`g^{(2)}(0)`
    :param indistinguishability: 2-photon mean wavepacket overlap
    :param losses: optical losses
    :param multiphoton_model: `distinguishable` if additional photons are distinguishable, `indistinguishable` otherwise
    :param context: gives a local context for source specific features, like `discernability_tag`
    """

    def __init__(self,
                 emission_probability: float = 1,
                 multiphoton_component: float = 0,
                 indistinguishability: float = 1,
                 losses: float = 0,
                 multiphoton_model: str = DISTINGUISHABLE_KEY,  # Literal[DISTINGUISHABLE_KEY, INDISTINGUISHABLE_KEY]
                 context: dict = None) -> None:

        assert 0 < emission_probability <= 1, "emission_probability must be in ]0;1]"
        assert 0 <= losses <= 1, "losses must be in [0;1]"
        assert 0 <= multiphoton_component <= 1, "multiphoton_component must be in [0;1]"
        assert emission_probability * multiphoton_component <= 0.5,\
            "emission_probability * g2 higher than 0.5 can not be computed for now"
        assert multiphoton_model in [DISTINGUISHABLE_KEY, INDISTINGUISHABLE_KEY], \
            "invalid value for multiphoton_model"

        self._emission_probability = emission_probability
        self._losses = losses
        self._multiphoton_component = multiphoton_component
        self._multiphoton_model = multiphoton_model
        self._indistinguishability = indistinguishability
        self._context = context or {}
        if "discernability_tag" not in self._context:
            self._context["discernability_tag"] = 0

        self.simplify_distribution = False  # Simplify the distribution by anonymizing photon annotations (can be
                                             # time-consuming for larger distributions)

    @staticmethod
    def from_noise_model(noise: NoiseModel):
        if noise is None:
            return Source()
        return Source(emission_probability=noise.brightness,
                      multiphoton_component=noise.g2,
                      indistinguishability=noise.indistinguishability,
                      losses=1 - noise.transmittance,
                      multiphoton_model=DISTINGUISHABLE_KEY if noise.g2_distinguishable else INDISTINGUISHABLE_KEY)

    def get_tag(self, tag, add=False):
        if add:
            self._context[tag] += 1
        return self._context[tag]

    def _get_probs(self):
        px = self._emission_probability
        g2 = self._multiphoton_component
        eta = 1 - self._losses

        # Starting formulas
        # g2 = 2p2/(p1+2p2)**2
        # p1 + p2 + ... = px & pn<<p2 for n>2

        p2 = (- px * g2 - math.sqrt(1 - 2 * px * g2) + 1) / g2 if g2 else 0
        p1 = px - p2

        p1to1 = eta * p1
        p2to2 = eta ** 2 * p2
        p2to1 = eta * (1 - eta) * p2
        return p1to1, p2to1, p2to2

    @staticmethod
    def _add(plist: BSDistribution, annotations: int | list, probability: float):
        # Add an annotation list (or a number of unannotated photons) and its probability to the in/out
        # parameter `plist`
        if probability > 0:
            if isinstance(annotations, int):
                plist[BasicState([annotations])] = probability
                return
            plist[BasicState([len(annotations)], {0: annotations})] = probability

    def _generate_one_photon_distribution(self) -> BSDistribution:
        # Generates a distribution of annotations given the source parameters for one photon in one mode
        distinguishability = 1 - math.sqrt(self._indistinguishability)

        # Approximation distinguishable photons are pure
        distinguishable_photon = self.get_tag("discernability_tag", add=True)
        second_photon = self.get_tag("discernability_tag", add=True) \
            if self._multiphoton_model == DISTINGUISHABLE_KEY else 0  # Noise photon or signal

        (p1to1, p2to1, p2to2) = self._get_probs()
        p0 = 1 - (p1to1 + 2 * p2to1 + p2to2)  # 2 * p2to1 because of symmetry

        dist = BSDistribution()
        self._add(dist, 0, p0)
        if self.partially_distinguishable:
            self._add(dist, ["_:0", "_:%s" % second_photon], (1 - distinguishability) * p2to2)
            self._add(dist, ["_:%s" % distinguishable_photon, "_:%s" % second_photon], distinguishability * p2to2)
            if self._multiphoton_model == DISTINGUISHABLE_KEY:
                self._add(dist, ["_:%s" % distinguishable_photon], distinguishability * (p1to1 + p2to1) + p2to1)
                self._add(dist, ["_:0"], (1 - distinguishability) * (p1to1 + p2to1))
            else:
                self._add(dist, ["_:%s" % distinguishable_photon], distinguishability * (p1to1 + p2to1))
                self._add(dist, ["_:0"], (1 - distinguishability) * (p1to1 + p2to1) + p2to1)
        else:
            # Just avoids annotations
            self._add(dist, 2, p2to2)
            self._add(dist, 1, p1to1 + 2 * p2to1)
        return dist

    @property
    def partially_distinguishable(self):
        return self._indistinguishability != 1 \
            or (self._multiphoton_model == DISTINGUISHABLE_KEY and self._multiphoton_component)

    def probability_distribution(self, nphotons: int = 1, prob_threshold: float = 0) -> SVDistribution:
        r"""returns SVDistribution on 1 mode associated to the source

        :param nphotons: Require `nphotons` in the mode (default 1).
        :param prob_threshold: Probability threshold under which the resulting state is filtered out.
        """
        if nphotons == 0 or self.is_perfect():
            return SVDistribution(StateVector([nphotons]))
        dist_all = BSDistribution.list_tensor_product([self._generate_one_photon_distribution() for _ in range(nphotons)],
                                                      merge_modes=True, prob_threshold=prob_threshold)

        return SVDistribution(dist_all)

    def generate_distribution(self, expected_input: BasicState, prob_threshold: float = 0):
        """
        Simulates plugging the photonic source on certain modes and turning it on.
        Computes the input probability distribution

        :param expected_input: Expected input BasicState
            The properties of the source will alter the input state. A perfect source always delivers the expected state
            as an input. Imperfect ones won't.
        :param prob_threshold: Probability threshold under which the resulting state is filtered out. By default,
            `global_params['min_p']` value is used.
        """
        prob_threshold = max(prob_threshold, global_params['min_p'])
        get_logger().debug(f"Apply 'Source' noise model to {expected_input}", channel.general)

        distributions = [self.probability_distribution(photon_count, prob_threshold) for photon_count in expected_input]
        dist = SVDistribution.list_tensor_product(distributions, prob_threshold=prob_threshold)

        dist.normalize()
        if self.simplify_distribution and self.partially_distinguishable:
            dist = anonymize_annotations(dist, annot_tag='_')
        return dist

    def is_perfect(self) -> bool:
        return \
            self._emission_probability == 1 and \
            self._multiphoton_component == 0 and \
            self._indistinguishability == 1 and \
            self._losses == 0

    def __eq__(self, value: object) -> bool:
        return \
            self._emission_probability == value._emission_probability and \
            self._losses == value._losses and \
            self._multiphoton_component == value._multiphoton_component and \
            self._multiphoton_model == value._multiphoton_model and \
            self._indistinguishability == value._indistinguishability and \
            self._context == value._context and \
            self.simplify_distribution == value.simplify_distribution

    def __dict__(self) -> dict:
        return {'g2': self._multiphoton_component,
                'transmittance': 1 - self._losses,
                'brightness': self._emission_probability,
                'indistinguishability': self._indistinguishability,
                'g2_distinguishable': self._multiphoton_model == DISTINGUISHABLE_KEY}
