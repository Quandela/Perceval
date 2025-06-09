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

from exqalibur import BSSamples
import exqalibur as xq

from perceval.utils import SVDistribution, BasicState, anonymize_annotations, NoiseModel, global_params
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

        if context is None:
            tag = 0
        else:
            tag = context.get("discernability_tag", 0)

        self._source = xq.Source(emission_probability,
                                 multiphoton_component,
                                 indistinguishability,
                                 losses,
                                 multiphoton_model == DISTINGUISHABLE_KEY,
                                 tag)

        self.simplify_distribution = False  # Simplify the distribution by anonymizing photon annotations (can be
                                             # time-consuming for larger distributions)

        self._sampler : xq.SourceSampler | None = None

    @staticmethod
    def from_noise_model(noise: NoiseModel):
        if noise is None:
            return Source()
        return Source(emission_probability=noise.brightness,
                      multiphoton_component=noise.g2,
                      indistinguishability=noise.indistinguishability,
                      losses=1 - noise.transmittance,
                      multiphoton_model=DISTINGUISHABLE_KEY if noise.g2_distinguishable else INDISTINGUISHABLE_KEY)
    #
    # def get_tag(self, tag, add=False):
    #     if add:
    #         self._context[tag] += 1
    #     return self._context[tag]
    #
    # def _get_probs(self):
    #     px = self._emission_probability
    #     g2 = self._multiphoton_component
    #     eta = 1 - self._losses
    #
    #     # Starting formulas
    #     # g2 = 2p2/(p1+2p2)**2
    #     # p1 + p2 + ... = px & pn<<p2 for n>2
    #
    #     p2 = (- px * g2 - math.sqrt(1 - 2 * px * g2) + 1) / g2 if g2 else 0
    #     p1 = px - p2
    #
    #     p1to1 = eta * p1
    #     p2to2 = eta ** 2 * p2
    #     p2to1 = eta * (1 - eta) * p2
    #     return p1to1, p2to1, p2to2
    #
    # @staticmethod
    # def _add(plist: BSDistribution, annotations: int | list, probability: float):
    #     # Add an annotation list (or a number of unannotated photons) and its probability to the in/out
    #     # parameter `plist`
    #     if probability > 0:
    #         if isinstance(annotations, int):
    #             plist[BasicState([annotations])] = probability
    #             return
    #         plist[BasicState([len(annotations)], {0: annotations})] = probability
    #
    # def _generate_one_photon_distribution(self) -> BSDistribution:
    #     # Generates a distribution of annotations given the source parameters for one photon in one mode
    #     distinguishability = 1 - math.sqrt(self._indistinguishability)
    #
    #     # Approximation distinguishable photons are pure
    #     distinguishable_photon = self.get_tag("discernability_tag", add=True)
    #     second_photon = self.get_tag("discernability_tag", add=True) \
    #         if self._multiphoton_model == DISTINGUISHABLE_KEY else 0  # Noise photon or signal
    #
    #     (p1to1, p2to1, p2to2) = self._get_probs()
    #     p0 = 1 - (p1to1 + 2 * p2to1 + p2to2)  # 2 * p2to1 because of symmetry
    #
    #     dist = BSDistribution()
    #     self._add(dist, 0, p0)
    #     if self.partially_distinguishable:
    #         self._add(dist, ["_:0", "_:%s" % second_photon], (1 - distinguishability) * p2to2)
    #         self._add(dist, ["_:%s" % distinguishable_photon, "_:%s" % second_photon], distinguishability * p2to2)
    #         if self._multiphoton_model == DISTINGUISHABLE_KEY:
    #             self._add(dist, ["_:%s" % distinguishable_photon], distinguishability * (p1to1 + p2to1) + p2to1)
    #             self._add(dist, ["_:0"], (1 - distinguishability) * (p1to1 + p2to1))
    #         else:
    #             self._add(dist, ["_:%s" % distinguishable_photon], distinguishability * (p1to1 + p2to1))
    #             self._add(dist, ["_:0"], (1 - distinguishability) * (p1to1 + p2to1) + p2to1)
    #     else:
    #         # Just avoids annotations
    #         self._add(dist, 2, p2to2)
    #         self._add(dist, 1, p1to1 + 2 * p2to1)
    #     return dist

    @property
    def partially_distinguishable(self):
        return self._source.partially_distinguishable

    @property
    def emission_probability(self):
        return self._source.emission_probability

    @property
    def multiphoton_component(self):
        return self._source.multiphoton_component

    @property
    def indistinguishability(self):
        return self._source.indistinguishability

    @property
    def losses(self):
        return self._source.losses

    @property
    def multiphoton_model(self):
        return DISTINGUISHABLE_KEY if self._source.is_g2_distinguishable else INDISTINGUISHABLE_KEY

    def probability_distribution(self, nphotons: int = 1, prob_threshold: float = 0) -> SVDistribution:
        r"""returns SVDistribution on 1 mode associated to the source

        :param nphotons: Require `nphotons` in the mode (default 1).
        :param prob_threshold: Probability threshold under which the resulting state is filtered out.
        """
        return self._source.probability_distribution(nphotons, prob_threshold)

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

        dist = self._source.generate_distribution(expected_input, prob_threshold)

        if self.simplify_distribution and self.partially_distinguishable:
            dist = anonymize_annotations(dist, annot_tag='_')
        return dist

    def cache_prob_table(self, n: int, min_photons_filter: int = 0) -> tuple[float, float]:
        """
        Computes the prob_table. Removes the events having less than min_photons_filter photons.
        Cache the result.

        :return: the physical performance and the zero-photon probability
        """

        self._sampler = self.create_sampler(n, min_photons_filter)
        return self._sampler.physical_perf, self._sampler.zpp

    def create_sampler(self, n: int, min_photons_filter: int = 0) -> xq.SourceSampler:
        return self._source.create_sampler(n, min_photons_filter)

    # def _generate_samples_no_filter(self, max_samples: int, expected_input: BasicState) -> BSSamples:
    #     """
    #     Generate samples directly from the source, without generating the source probability distribution first.
    #
    #     :param max_samples: Number of samples to generate
    #     :param expected_input: Expected input BasicState
    #         The properties of the source will alter the input state. A perfect source always delivers the expected state
    #         as an input. Imperfect ones won't.
    #     """
    #     samples = BSSamples()
    #
    #     if not self.partially_distinguishable:
    #         bsd = self._generate_one_photon_distribution()
    #
    #     for photon_count in expected_input:
    #         new_samples = BSSamples()
    #         if photon_count == 0:
    #             new_samples.extend([BasicState([photon_count])] * max_samples)
    #         else:
    #             for _ in range(photon_count):
    #                 if self.partially_distinguishable:
    #                     bsd = self._generate_one_photon_distribution()
    #                 new_samples_one_mode = bsd.sample(max_samples, non_null=False)
    #                 if len(new_samples) == 0:
    #                     new_samples = new_samples_one_mode # first samples
    #                     continue
    #                 for i in range(len(new_samples_one_mode)):
    #                     new_samples[i] = new_samples[i].merge(new_samples_one_mode[i])
    #         if len(samples) == 0:
    #             samples = new_samples # first samples
    #             continue
    #         for i in range(len(new_samples)):
    #             samples[i] *= new_samples[i]
    #
    #     return samples

    def generate_samples(self, max_samples: int, expected_input: BasicState, min_detected_photons = 0) -> BSSamples:
        if self.is_perfect():
            return BSSamples([expected_input] * max_samples)

        # if min_detected_photons == 0:
        #     return self._generate_samples_no_filter(max_samples, expected_input)

        transmission = self.emission_probability * (1 - self.losses)
        if transmission == 0 and min_detected_photons >= 1:
            get_logger().warn(f"No useful state will be computed, aborting", channel.user)
            return BSSamples()

        if self._sampler is None or expected_input.n != self._sampler.n or min_detected_photons != self._sampler.min_photons_filter:
            self._sampler = self.create_sampler(expected_input.n, min_detected_photons)

        self._sampler.expected_input = expected_input

        return self._sampler.generate_samples(max_samples)

    def is_perfect(self) -> bool:
        return self._source.is_perfect

    def __eq__(self, value: Source) -> bool:
        return self._source == value._source and self.simplify_distribution == value.simplify_distribution

    def __dict__(self) -> dict:
        return {'g2': self.multiphoton_component,
                'transmittance': 1 - self.losses,
                'brightness': self.emission_probability,
                'indistinguishability': self.indistinguishability,
                'g2_distinguishable': self._source.is_g2_distinguishable}
