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
import numpy as np

from perceval.utils import SVDistribution, StateVector
from typing import Dict, Literal


class Source:
    r"""Definition of a source

            :param brightness: the probability per laser pulse to emmit at least one photon. Independent of all losses.
            :param multiphoton_component: second order intensity autocorrelation at zero time delay :math:`g^{(2)}(0)`
            :param multiphoton_model: `distinguishable` if additional photons are distinguishable, `indistinguishable`
              otherwise
            :param purity: preserved for back-compatibility if multiphoton_model is not set.`
            :param indistinguishability: indistinguishability parameter as defined by `indistinguishability_model`
            :param indistinguishability_model: `homv` defines indistinguishability as 2-photon wavepacket overlap,
                `linear` defines indistinguishability as ratio of indistinguishable photons
            :param overall_transmission: transmission of the optical system.
            :param context: gives a local context for source specific features, like `discernability_tag`
            """

    def __init__(self, brightness: float = 1,
                 multiphoton_component: float = None,
                 multiphoton_model: Literal["distinguishable", "indistinguishable"] = "distinguishable",
                 purity: float = None,
                 indistinguishability: float = 1,
                 indistinguishability_model: Literal["homv", "linear"] = "homv",
                 overall_transmission: float = 1,
                 context: Dict = None) -> None:

        if multiphoton_component is None:
            if purity is None:
                multiphoton_component = 0
            else:
                p2 = brightness * (1 - purity)
                p1 = brightness - p2
                multiphoton_component = 2 * p2 / (p1 + 2 * p2) ** 2
        else:
            assert purity is None, "cannot set both purity and multiphoton_component"
        assert brightness * multiphoton_component <= 0.5, "brightness * g2 higher than 0.5 can not be computed for now"
        self.brightness = brightness
        self.overall_transmission = overall_transmission
        self.multiphoton_component = multiphoton_component
        self._multiphoton_model = multiphoton_model
        assert self._multiphoton_model in ["distinguishable", "indistinguishable"], "invalid value for purity_model"
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
        g2 = self.multiphoton_component
        eta = self.overall_transmission
        beta = self.brightness

        # Starting formulas
        # g2 = 2p2/(p1+2p2)**2
        # p1 + p2 = beta

        # p2 = min(np.poly1d([g2, -2 * (1 - g2 * beta), g2 * beta ** 2]).r)
        p2 = (- beta * g2 - math.sqrt(1 - 2 * beta * g2) + 1) / g2 if g2 else 0
        p1 = beta - p2

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
