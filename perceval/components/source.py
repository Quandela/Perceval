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
    def __init__(self, brightness: float = 1,
                 purity: float = 1,
                 purity_model: Literal["random", "indistinguishable"] = "random",
                 indistinguishability: float = 1,
                 indistinguishability_model: Literal["homv", "linear"] = "homv",
                 context: Dict = None) -> None:
        r"""Definition of a source

        :param brightness: the brightness of the source defined as the percentage of unique photon generation
        :param purity: the ratio of time when photon is emitted alone
        :param purity_model: `random` if additional photons are distinguishable, `indistinguishable` otherwise
        :param indistinguishability: indistinguishability parameter as defined by `indistinguishability_model`
        :param indistinguishability_model: `homv` defines indistinguishability as HOM visibility, `linear` defines
            indistinguishability as ratio of indistinguishable photons
        :param context: gives a local context for source specific features, like `discernability_tag`
        """
        self.brightness = brightness
        self.purity = purity
        self._purity_model = purity_model
        self.indistinguishability = indistinguishability
        self._indistinguishability_model = indistinguishability_model
        assert self._indistinguishability_model in ["homv", "linear"], "invalid value for indistinguishability_model"
        self._context = context or {}
        if "discernability_tag" not in self._context:
            self._context["discernability_tag"] = 1

    def probability_distribution(self):
        r"""returns SVDistribution on 1 mode associated to the source
        """
        svd = SVDistribution()
        if self.brightness != 1:
            svd[StateVector([0])] = 1-self.brightness
        if self._indistinguishability_model == "homv":
            distinguishability = 1-math.sqrt(self.indistinguishability)
        else:
            distinguishability = 1 - self.indistinguishability
        # Approximation distinguishable photons are pure
        if self.purity != 1:
            if distinguishability:
                if self._purity_model == "random":
                    random_feat = self._context["discernability_tag"]
                    svd[StateVector([2], {1: {"_": 0}, 2: {"_": random_feat}})] = self.brightness * (1 - self.purity)
                    self._context["discernability_tag"] += 1
                else:
                    svd[StateVector([2], {1: {"_": 0}, 2: {"_": 0}})] = self.brightness * (1 - self.purity)
            else:
                svd[StateVector([2])] = self.brightness*(1-self.purity)
        if distinguishability:
            random_feat = self._context["discernability_tag"]
            self._context["discernability_tag"] += 1
            svd[StateVector([1], {1: {"_": random_feat}})] = distinguishability*self.brightness*self.purity
            svd[StateVector([1], {1: {"_": 0}})] = (1-distinguishability)*self.brightness*self.purity
        else:
            svd[StateVector([1])] = (1-distinguishability)*self.brightness*self.purity

        return svd
