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

    :param emission_probability: probability that the source emits at least one photon
    :param multiphoton_component: second order intensity autocorrelation at zero time delay :math:`g^{(2)}(0)`
    :param indistinguishability: 2-photon mean wavepacket overlap
    :param losses: optical losses
    :param multiphoton_model: `distinguishable` if additional photons are distinguishable, `indistinguishable` otherwise
    :param context: gives a local context for source specific features, like `discernability_tag`
    """

    def __init__(self, emission_probability: float = 1,
                 multiphoton_component: float = 0,
                 indistinguishability: float = 1,
                 losses: float = 0,
                 multiphoton_model: Literal["distinguishable", "indistinguishable"] = "distinguishable",
                 context: Dict = None) -> None:

        assert 0 < emission_probability <= 1, "emission_probability must be in ]0;1]"
        assert 0 <= losses <= 1, "losses must be in [0;1]"
        assert 0 <= multiphoton_component <= 1, "multiphoton_component must be in [0;1]"
        assert emission_probability * multiphoton_component <= 0.5,\
            "emission_probability * g2 higher than 0.5 can not be computed for now"
        assert multiphoton_model in ["distinguishable", "indistinguishable"], \
            "invalid value for multiphoton_model"

        self._emission_probability = emission_probability
        self._losses = losses
        self._multiphoton_component = multiphoton_component
        self._multiphoton_model = multiphoton_model
        self._indistinguishability = indistinguishability
        self._context = context or {}
        if "discernability_tag" not in self._context:
            self._context["discernability_tag"] = 0

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

    def probability_distribution(self):
        r"""returns SVDistribution on 1 mode associated to the source
        """
        distinguishability = 1 - math.sqrt(self._indistinguishability)

        # Approximation distinguishable photons are pure
        distinguishable_photon = self.get_tag("discernability_tag", add=True)
        second_photon = self.get_tag("discernability_tag", add=True) \
            if self._multiphoton_model == "distinguishable" else 0  # Noise photon or signal

        (p1to1, p2to1, p2to2) = self._get_probs()

        svd = SVDistribution()

        # 2 * p2to1 because of symmetry
        p0 = 1 - (p1to1 + 2 * p2to1 + p2to2)
        svd.add(StateVector([0]), p0)

        if distinguishability or (self._multiphoton_model == "distinguishable" and p2to2):
            svd.add(StateVector([2], {0: ["_: 0", "_:%s" % second_photon]}),
                    (1 - distinguishability) * p2to2)
            svd.add(StateVector([2], {0: ["_:%s" % distinguishable_photon, "_:%s" % second_photon]}),
                    distinguishability * p2to2)
            svd.add(StateVector([1], {0: ["_:%s" % distinguishable_photon]}),
                    distinguishability * (p1to1 + p2to1))
            svd.add(StateVector([1], {0: ["_:0"]}),
                    (1 - distinguishability) * (p1to1 + p2to1))
            svd.add(StateVector([1], {0: ["_:%s" % second_photon]}),
                    p2to1)
        else:
            # Just avoids annotations
            svd.add(StateVector([2]), p2to2)
            svd.add(StateVector([1]), p1to1 + 2 * p2to1)

        return svd
