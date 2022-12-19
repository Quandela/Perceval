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
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import math

from perceval.utils import SVDistribution, StateVector
from typing import Dict, Literal


class Source:
    r"""Definition of a source
    We build on a phenomenological model first introduced in ref. [1] where an imperfect quantum-dot based single-photon
    source is modeled by a statistical mixture of Fock states. The model developed here, first introduced in ref. [2],
    constructs the input multi-photon state using features specific to Perceval.
    [1] Pont, Mathias, et al. Physical Review X 12, 031033 (2022). https://doi.org/10.1103/PhysRevX.12.031033
    [2] Pont, Mathias, et al. arXiv preprint arXiv:2211.15626 (2022). https://doi.org/10.48550/arXiv.2211.15626

    :param occupation_factor: occupation factor of the QD state.
    :param multiphoton_component: second order intensity autocorrelation at zero time delay :math:`g^{(2)}(0)`
    :param multiphoton_model: `distinguishable` if additional photons are distinguishable, `indistinguishable` otherwise
    :param indistinguishability: indistinguishability parameter as defined by 'indistinguishability_model'
    :param indistinguishability_model: `homv` defines indistinguishability as 2-photon mean wavepacket overlap,
        `linear` defines indistinguishability as ratio of indistinguishable photons
    :param brightness: Number of photons collected per excitation pulse into the first lens.
    :param overall_transmission: Total transmission of the optical system. Can take into account the brightness.
    :param context: gives a local context for source specific features, like `discernability_tag`
    """

    def __init__(self, occupation_factor: float = 1,
                 multiphoton_component: float = 0,
                 multiphoton_model: Literal["distinguishable", "indistinguishable"] = "distinguishable",
                 indistinguishability: float = 1,
                 indistinguishability_model: Literal["homv", "linear"] = "homv",
                 brightness: float = 1,
                 overall_transmission: float = 1,
                 context: Dict = None) -> None:

        assert brightness * multiphoton_component <= 0.5, "brightness * g2 higher than 0.5 can not be computed for now"
        self.brightness = brightness
        self.occupation_factor = occupation_factor
        self.overall_transmission = overall_transmission
        # By definition brightness=beta*eta_out*px where beta is the fraction of emission into the mode and eta_out is
        # the out-coupling efficiency
        assert brightness * overall_transmission < occupation_factor, "Set of parameters is not physically acceptable"
        self.multiphoton_component = multiphoton_component
        self._multiphoton_model = multiphoton_model
        assert self._multiphoton_model in ["distinguishable", "indistinguishable"], "invalid value for multiphoton_model"
        self.indistinguishability = indistinguishability
        self._indistinguishability_model = indistinguishability_model
        assert self._indistinguishability_model in ["homv", "linear"], "invalid value for indistinguishability_model"
        self._context = context or {}
        if "discernability_tag" not in self._context:
            self._context["discernability_tag"] = 0

    def get_tag(self, tag, add=False):
        if add:
            self._context[tag] += 1
        return self._context[tag]

    def _get_probs(self):
        px = self.occupation_factor
        g2 = self.multiphoton_component
        eta = self.overall_transmission*self.brightness/self.occupation_factor

        # Starting formulas
        # g2 = 2p2/(p1+2p2)**2
        # p1 + p2 + ... = px & pn<<p2 for n>2

        p2 = (- px * g2 - math.sqrt(1 - 2 * px * g2) + 1) / g2 if g2 else 0
        p1 = px - p2

        p1to1 = eta * p1
        p2to2 = eta ** 2 * p2
        p2to1 = eta * (1 - eta) * p2

        return p1to1, p2to1, p2to2

    def probability_distribution(self):
        r"""returns SVDistribution on 1 mode associated to the source
        """
        # states with probability 0 will be removed by processor

        if self._indistinguishability_model == "homv":
            distinguishability = 1-math.sqrt(self.indistinguishability)
        else:
            distinguishability = 1-self.indistinguishability

        # Approximation distinguishable photons are pure
        distinguishable_photon = self.get_tag("discernability_tag", add=True)
        second_photon = self.get_tag("discernability_tag", add=True) \
            if self._multiphoton_model == "distinguishable" else 0  # Noise photon or signal

        (p1to1, p2to1, p2to2) = self._get_probs()

        svd = SVDistribution()

        # 2 * p2to1 because of symmetry
        p0 = 1 - (p1to1 + 2 * p2to1 + p2to2)
        svd[StateVector([0])] = p0

        if distinguishability or (self._multiphoton_model == "distinguishable" and p2to2):
            svd[StateVector([2], {0: ["_: 0", "_:%s" % second_photon]})] = (1 - distinguishability) * p2to2
            svd[StateVector([2], {0: ["_:%s" % distinguishable_photon,
                                      "_:%s" % second_photon]})] = distinguishability * p2to2
            svd[StateVector([1], {0: ["_:%s" % distinguishable_photon]})] = distinguishability * (p1to1 + p2to1)
            svd[StateVector([1], {0: ["_:0"]})] = (1 - distinguishability) * (p1to1 + p2to1)
            svd[StateVector([1], {0: ["_:%s" % second_photon]})] += p2to1
        else:
            # Just avoids annotations
            svd[StateVector([2])] = p2to2
            svd[StateVector([1])] = p1to1 + 2 * p2to1

        return svd
