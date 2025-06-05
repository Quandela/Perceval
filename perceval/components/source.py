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
import random

from itertools import repeat

from exqalibur import BSSamples

from perceval.utils import (SVDistribution, StateVector, BasicState, anonymize_annotations, NoiseModel, global_params,
                            BSDistribution, FSIDistribution)
from perceval.utils.logging import get_logger, channel
from perceval.utils.statevector import FockStateIndex

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

    FSD_TYPE = FSIDistribution
    FS_TYPE = FockStateIndex

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
        self._prob_table = None
        self._prob_table_n = None
        self._prob_table_filter = None

        self._single_photon_dist_cache: Source.FSD_TYPE | None = None

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

    def _compute_prob_table(self, n: int, min_photons_filter: int = 0) -> tuple[dict[tuple[int, int, int], float], float, float]:
        """
        Computes the table of probability of getting (i, j, k) events from n wanted photons where
        i is the number of single signal photon events
        j is the number of single g2 photon events
        k is the number of signal + g2 photons events

        Distinguishable photons are still counted as signal.

        :param n: Number of photons wanted.
        :param min_photons_filter: Minimum number of photons wanted.

        :return: the probability table, the physical performance, and the zero-photon probability
        """

        p1to1, p2to1, p2to2 = self._get_probs()
        p_signal = p1to1 + p2to1
        p_g2 = p2to1
        p_duo = p2to2
        p0 = 1 - (p_signal + p_g2 + p_duo)

        factorial_table = {0: 1}
        def cache_factorial(m):
            if m not in factorial_table:
                factorial_table[m] = m * cache_factorial(m - 1)
            return factorial_table[m]

        prob_table = {}

        fac_n = cache_factorial(n)  # Fills the table for all needed values
        for i in range(n + 1):
            fac_i = factorial_table[i]
            for j in range(n + 1 - i if p_g2 else 1):
                fac_j = factorial_table[j]
                for k in range(n + 1 - i - j if p_duo else 1):
                    fac_k = factorial_table[k]
                    if i + j + 2 * k >= min_photons_filter:
                        # TODO: remove low probability events ?
                        n0 = n - i - j - k
                        prob_table[(i, j, k)] = (fac_n * p_signal ** i * p_g2 ** j * p_duo ** k * p0 ** n0
                                                 / (fac_i * fac_j * fac_k * factorial_table[n0]))

        phys_perf = sum(prob_table.values())
        if min_photons_filter:
            for key, prob in prob_table.items():
                prob_table[key] = prob / phys_perf

        return prob_table, phys_perf, p0 ** n

    def cache_prob_table(self, n: int, min_photons_filter: int = 0) -> tuple[float, float]:
        """
        Computes the prob_table. Removes the events having less than min_photons_filter photons.
        Cache the result.

        :return: the physical performance and the zero-photon probability
        """

        prob_table, phys_perf, zpp = self._compute_prob_table(n, min_photons_filter)
        self._prob_table = prob_table
        self._prob_table_n = n
        self._prob_table_filter = min_photons_filter
        return phys_perf, zpp

    def _generate_distinguishability(self, n: int):
        """Generate a random list of booleans of size n such that False means that a given photon is distinguishable
        and True means that a given photon is indistinguishable."""
        indistinguishability = math.sqrt(self._indistinguishability)

        return random.choices([True, False], k=n, weights=[indistinguishability, 1 - indistinguishability])

    def _events_to_samples(self, events: list[tuple[int, int, int]], expected_input: BasicState):
        res = BSSamples()
        dist_index = 0
        dist_list = self._generate_distinguishability(sum(event[0] + event[2] for event in events))

        first_tag = self.get_tag("discernability_tag")  # Just to avoid growing up too much the complex that represents the tag

        empty_bs = BasicState([0])
        signal_state = BasicState("|{_:0}>")  # Avoids creating many times the same states
        expected_input_list = list(expected_input)

        # TODO: parallelize this?
        for event in events:
            photons = []
            for _ in range(event[0]):
                # signal alone
                if dist_list[dist_index]:
                    photons.append(signal_state)
                else:
                    photons.append(BasicState([1], {0: [f"_:{self.get_tag('discernability_tag', add=True)}"]}))
                dist_index += 1

            for _ in range(event[1]):
                # g2 alone
                second_photon = self.get_tag("discernability_tag", add=True) \
                    if self._multiphoton_model == DISTINGUISHABLE_KEY else 0  # Noise photon or signal
                photons.append(BasicState([1], {0: [f"_:{second_photon}"]}))

            for _ in range(event[2]):
                # signal + g2
                first_photon = 0 if dist_list[dist_index] else self.get_tag("discernability_tag", add=True)
                second_photon = self.get_tag("discernability_tag", add=True) \
                    if self._multiphoton_model == DISTINGUISHABLE_KEY else 0  # Noise photon or signal
                photons.append(BasicState([2], {0: [f"_:{first_photon}", f"_:{second_photon}"]}))
                dist_index += 1

            photons += [empty_bs] * (expected_input.n - len(photons))
            random.shuffle(photons)

            index = 0
            final_state = BasicState()
            for n_photons in expected_input_list:
                single_mode_state = empty_bs
                for _ in range(n_photons):
                    if single_mode_state.n:
                        single_mode_state = single_mode_state.merge(photons[index])
                    else:
                        single_mode_state = photons[index]
                    index += 1

                if len(final_state):
                    final_state *= single_mode_state
                else:
                    final_state = single_mode_state

            res.append(final_state)
            self._context["discernability_tag"] = first_tag

        return res

    def _generate_samples_no_filter(self, max_samples: int, expected_input: BasicState) -> BSSamples:
        """
        Generate samples directly from the source, without generating the source probability distribution first.

        :param max_samples: Number of samples to generate
        :param expected_input: Expected input BasicState
            The properties of the source will alter the input state. A perfect source always delivers the expected state
            as an input. Imperfect ones won't.
        """
        samples = BSSamples()

        if not self.partially_distinguishable:
            bsd = self._generate_one_photon_distribution()

        for photon_count in expected_input:
            new_samples = BSSamples()
            if photon_count == 0:
                new_samples.extend([BasicState([photon_count])] * max_samples)
            else:
                for _ in range(photon_count):
                    if self.partially_distinguishable:
                        bsd = self._generate_one_photon_distribution()
                    new_samples_one_mode = bsd.sample(max_samples, non_null=False)
                    if len(new_samples) == 0:
                        new_samples = new_samples_one_mode # first samples
                        continue
                    for i in range(len(new_samples_one_mode)):
                        new_samples[i] = new_samples[i].merge(new_samples_one_mode[i])
            if len(samples) == 0:
                samples = new_samples # first samples
                continue
            for i in range(len(new_samples)):
                samples[i] *= new_samples[i]

        return samples

    def generate_samples(self, max_samples: int, expected_input: BasicState, min_detected_photons = 0) -> BSSamples:
        if self.is_perfect():
            return BSSamples([expected_input] * max_samples)

        if min_detected_photons == 0:
            return self._generate_samples_no_filter(max_samples, expected_input)

        transmission = self._emission_probability * (1 - self._losses)
        if transmission == 0 and min_detected_photons >= 1:
            get_logger().warn(f"No useful state will be computed, aborting", channel.user)
            return BSSamples()

        if self._prob_table is None or expected_input.n != self._prob_table_n or min_detected_photons != self._prob_table_filter:
            self.cache_prob_table(expected_input.n, min_detected_photons)

        events = random.choices(list(self._prob_table.keys()), k=max_samples, weights=self._prob_table.values())
        return self._events_to_samples(events, expected_input)

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


    @classmethod
    def _add_st(cls, plist: FSD_TYPE, signal: int, probability: float):
        # Add an annotation list (or a number of unannotated photons) and its probability to the in/out
        # parameter `plist`
        if probability > 0:
            plist[cls.FS_TYPE((signal,))] = probability

    # TODO: we can cache this so it is reused for all photons
    def _generate_one_photon_distribution_st(self):
        (p1to1, p2to1, p2to2) = self._get_probs()
        p0 = 1 - (p1to1 + 2 * p2to1 + p2to2)  # 2 * p2to1 because of symmetry

        dist = self.FSD_TYPE()
        self._add_st(dist, 0, p0)

        # Just avoids annotations for now
        self._add_st(dist, 2, p2to2)
        self._add_st(dist, 1, p1to1 + 2 * p2to1)

        self._single_photon_dist_cache = dist

    def probability_distribution_st(self, nphotons: int = 1, prob_threshold: float = 0) -> FSD_TYPE:
        r"""returns SVDistribution on 1 mode associated to the source
        :param nphotons: Require `nphotons` in the mode (default 1).
        :param prob_threshold: Probability threshold under which the resulting state is filtered out.
        """
        # Note: we could cache this as the result for n photons in a single mode is now small
        if nphotons == 0 or self.is_perfect():
            return self.FSD_TYPE(self.FS_TYPE((nphotons,)))

        if self._single_photon_dist_cache is None:
            self._generate_one_photon_distribution_st()

        return self.FSD_TYPE.list_tensor_product([self._single_photon_dist_cache for _ in repeat(None, nphotons)],
                                                 merge_modes=True, prob_threshold=prob_threshold)

    def generate_state_distribution(self, expected_input: BasicState, prob_threshold: float = None) -> FSD_TYPE:
        """
        Simulates plugging the photonic source on certain modes and turning it on.
        Computes the input probability distribution
        :param expected_input: Expected input BasicState
            The properties of the source will alter the input state. A perfect source always delivers the expected state
            as an input. Imperfect ones won't.
        :param prob_threshold: Probability threshold under which the resulting state is filtered out. By default,
            `global_params['min_p']` value is used.
        """
        prob_threshold = global_params['min_p'] if prob_threshold is None else prob_threshold
        get_logger().debug(f"Apply 'Source' noise model to {expected_input}", channel.general)

        distributions = [self.probability_distribution_st(photon_count, prob_threshold) for photon_count in expected_input]
        return self.FSD_TYPE.list_tensor_product(distributions, prob_threshold=prob_threshold)
