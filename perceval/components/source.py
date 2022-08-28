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
    def __init__(self, brightness: float = 1,
                 multiphoton_component: float = None,
                 multiphoton_model: Literal["distinguishable", "indistinguishable"] = "distinguishable",
                 purity: float = None,
                 indistinguishability: float = 1,
                 indistinguishability_model: Literal["homv", "linear"] = "homv",
                 overall_transmission: float = 1,
                 context: Dict = None) -> None:
        r"""Definition of a source

        :param brightness: the probability per laser pulse to emmit at least one photon. Indenpendent of all losses.
        :param multiphoton_component: second order intensity autocorrelation at zero time delay :math:`g^{(2)}(0)`
        :param multiphoton_model: `distinguishable` if additional photons are distinguishable, `indistinguishable`
          otherwise
        :param purity: preserved for back-compatibility if multiphoton_model is not set. :math:`purity = 1-g^{(2)}(0)`
        :param indistinguishability: indistinguishability parameter as defined by `indistinguishability_model`
        :param indistinguishability_model: `homv` defines indistinguishability as 2-photon wavepacket overlap,
            `linear` defines indistinguishability as ratio of indistinguishable photons
        :param overall_transmission: transmission of the optical system.
        :param context: gives a local context for source specific features, like `discernability_tag`
        """
        if multiphoton_component is None:
            if purity is None:
                multiphoton_component = 0
            else:
                multiphoton_component = 1 - purity
        else:
            assert purity is None, "cannot set both purity and multiphoton_component"
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
            self._context["discernability_tag"] = 1

    def probability_distribution(self):
        r"""returns SVDistribution on 1 mode associated to the source
        """

        g2 = self.multiphoton_component
        eta = self.overall_transmission
        beta = self.brightness

        # g2 = 2p2/(p1+2p2)**2
        # p1 + p2 = beta

        p2 = min(np.poly1d([g2, -2 * (1 - g2 * beta), g2 * beta ** 2]).r)
        p1 = beta - p2

        svd = SVDistribution()

        svd[StateVector([0])] = 1-(eta*p1+eta**2*p2+2*eta*(1-eta)*p2)

        if self._indistinguishability_model == "homv":
            distinguishability = 1-math.sqrt(self.indistinguishability)
        else:
            distinguishability = 1-self.indistinguishability

        # Approximation distinguishable photons are pure
        distinguishable_photon = self._context["discernability_tag"]
        self._context["discernability_tag"] += 1
        noise_photon = self._context["discernability_tag"]
        self._context["discernability_tag"] += 1

        if p2 != 0:
            if distinguishability != 0:
                if self._multiphoton_model == "distinguishable":
                    svd[StateVector([2], {1: {"_": 0}, 2: {"_": noise_photon}})] = eta**2*(1-distinguishability)*p2
                    svd[StateVector([2], {1: {"_": distinguishable_photon}, 2: {"_": noise_photon}})] = eta**2*distinguishability*p2
                else:
                    svd[StateVector([2], {1: {"_": 0}, 2: {"_": 0}})] = eta**2*(1-distinguishability)*p2
                    svd[StateVector([2], {1: {"_": distinguishable_photon}, 2: {"_": 0}})] = eta**2*distinguishability*p2
            else:
                if self._multiphoton_model == "distinguishable":
                    svd[StateVector([2], {1: {"_": 0}, 2: {"_": noise_photon}})] = eta**2*p2
                else:
                    svd[StateVector([2])] = eta**2*p2

        if distinguishability != 0:
            if self._multiphoton_model == "distinguishable":
                svd[StateVector([1], {1: {"_": distinguishable_photon}})] = eta*distinguishability*p1+eta*(1-eta)*distinguishability*p2
                svd[StateVector([1], {1: {"_": 0}})] = eta*(1-distinguishability)*p1+eta*(1-eta)*(1-distinguishability)*p2
                svd[StateVector([1], {1: {"_": noise_photon}})] = eta*(1-eta)*p2
            else:
                svd[StateVector([1], {1: {"_": distinguishable_photon}})] = eta*distinguishability*p1+eta*(1-eta)*distinguishability*p2
                svd[StateVector([1], {1: {"_": 0}})] = eta*(1-distinguishability)*p1+eta*(1-eta)*(1-distinguishability)*p2+eta*(1-eta)*p2

        else:
            if p2 != 0:
                if self._multiphoton_model == "distinguishable":
                    svd[StateVector([1], {1: {"_": 0}})] = eta*p1+eta*(1-eta)*p2
                    svd[StateVector([1], {1: {"_": noise_photon}})] = eta*(1-eta)*p2
                else:
                    svd[StateVector([1])] = eta*p1+2*eta*(1-eta)*p2
            else:
                svd[StateVector([1])] = eta*p1

        return svd
