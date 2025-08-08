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

from perceval.utils import SVDistribution, NoisyFockState, FockState, anonymize_annotations, NoiseModel, global_params
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
    """

    def __init__(self,
                 emission_probability: float = 1,
                 multiphoton_component: float = 0,
                 indistinguishability: float = 1,
                 losses: float = 0,
                 multiphoton_model: str = DISTINGUISHABLE_KEY,  # Literal[DISTINGUISHABLE_KEY, INDISTINGUISHABLE_KEY]
                 ) -> None:

        self._source = xq.Source(emission_probability,
                                 multiphoton_component,
                                 indistinguishability,
                                 losses,
                                 multiphoton_model == DISTINGUISHABLE_KEY)

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
        return self.generate_distribution(FockState([nphotons]), prob_threshold)

    def generate_distribution(self, expected_input: FockState, prob_threshold: float = 0) -> SVDistribution:
        """
        Simulates plugging the photonic source on certain modes and turning it on.
        Computes the input probability distribution

        :param expected_input: Expected input FockState
            The properties of the source will alter the input state. A perfect source always delivers the expected state
            as an input. Imperfect ones won't.
        :param prob_threshold: Probability threshold under which the resulting state is filtered out. By default,
            `global_params['min_p']` value is used.
        """
        prob_threshold = max(prob_threshold, global_params['min_p'])
        get_logger().debug(f"Apply 'Source' noise model to {expected_input}", channel.general)

        dist = self._source.generate_distribution(expected_input, prob_threshold)

        # if self.simplify_distribution and self.partially_distinguishable:
        #     sv_dist = defaultdict(lambda: 0)
        #     for k, p in dist.items():
        #         sv_dist[k] += p
        #     return SVDistribution({k: v for k, v in sorted(sv_dist.items(), key=lambda x: -x[1])})

        # if self.simplify_distribution and self.partially_distinguishable:
        #     dist = anonymize_annotations(dist, annot_tag='_') # TODO : see if we still want an anonymize(NoisyFockState)
        return dist

    def create_iterator(self, expected_input: FockState, min_photons_filter: int = 0) -> xq.SimpleSourceIterator:
        """
        Creates a source iterator that can generate all already separated noisy states according
        to the probability distribution without representing them in memory.

        This is far more efficient than computing the whole distribution.

        Supports a min_photons_filter to avoid generating states having not enough photons.

        >>> from perceval import BasicState, Source
        >>>
        >>> source = Source(indistinguishability=0.85, losses=0.56)
        >>> iterator = source.create_iterator(BasicState([1, 0, 1]), 2)
        >>> iterator.prob_threshold = iterator.max_p * 1e-5  # Generates only states having at most 1e-5 times the biggest probability.
        >>> for separated_state, prob in iterator:
        >>>     print(separated_state, prob)

        :param expected_input: Expected input BasicState
        :param min_photons_filter: The minimum number of photons required to generate a state.
        """
        # TODO: chose iterator depending on the input state
        return self._source.create_simple_iterator(expected_input, min_photons_filter)

    def create_sampler(self, expected_input: FockState, min_photons_filter: int = 0) -> xq.SourceSampler:
        """
        Creates a source sampler that will be able to generate states according to the source probability distribution
        :param expected_input: The expected input BasicState to sample.
        :param min_photons_filter: Minimum number of photons in a sampled state
        """
        return self._source.create_sampler(expected_input, min_photons_filter)

    def generate_samples(self, max_samples: int, expected_input: FockState, min_detected_photons = 0) -> list[NoisyFockState] | list[FockState]:
        """
        Samples states from the source probability distribution without representing the whole distribution in memory.
        Creates a source sampler and store it in self for faster repeated sampling if necessary.

        :param max_samples: Number of samples to generate.
        :param expected_input: The nominal input state that the source should produce.
        :param min_detected_photons: Minimum number of photons in a sampled state.
        """
        if self.is_perfect():
            return [expected_input] * max_samples

        if self._sampler is None or min_detected_photons != self._sampler.min_photons_filter or expected_input != self._sampler.expected_input:
            self._sampler = self.create_sampler(expected_input, min_detected_photons)

        return self._sampler.generate_samples(max_samples)

    def generate_separated_samples(self, max_samples: int, expected_input: FockState, min_detected_photons = 0) -> list[BSSamples]:
        """
        Samples separated states from the source probability distribution without representing the whole distribution in memory.
        The sampled states are equivalent (up to permutation) to calling separate_state() on each state returned by generate_samples()
        but this sampling process uses a simplified procedure to do it faster.

        Creates a source sampler and store it in self for faster repeated sampling if necessary.

        :param max_samples: Number of samples to generate.
        :param expected_input: The nominal input state that the source should produce.
        :param min_detected_photons: Minimum number of photons in a sampled state.
        """
        if self.is_perfect():
            sample = BSSamples([expected_input])
            return [sample] * max_samples

        if self._sampler is None or min_detected_photons != self._sampler.min_photons_filter or expected_input != self._sampler.expected_input:
            self._sampler = self.create_sampler(expected_input, min_detected_photons)

        return self._sampler.generate_separated_samples(max_samples)

    def is_perfect(self) -> bool:
        return self._source.is_perfect

    def __eq__(self, value: Source) -> bool:
        return self._source == value._source and self.simplify_distribution == value.simplify_distribution

    def to_dict(self) -> dict:
        return {'g2': self.multiphoton_component,
                'transmittance': 1 - self.losses,
                'brightness': self.emission_probability,
                'indistinguishability': self.indistinguishability,
                'g2_distinguishable': self._source.is_g2_distinguishable}
