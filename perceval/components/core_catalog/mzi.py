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

from abc import ABC, abstractmethod
from perceval.components import Processor, Circuit, BS, PS
from perceval.components.component_catalog import CatalogItem

import numpy as np


class AMZI(CatalogItem, ABC):
    @staticmethod
    def _handle_params(**kwargs):
        if "i" in kwargs:
            kwargs["phi_a"] = f"phi_a{kwargs['i']}"
            kwargs["phi_b"] = f"phi_b{kwargs['i']}"
        return CatalogItem._handle_param(kwargs.get("phi_a", "phi_a")), \
            CatalogItem._handle_param(kwargs.get("phi_b", "phi_b")), \
            kwargs.get("theta_a", np.pi/2), \
            kwargs.get("theta_b", np.pi/2)

    def build_processor(self, **kwargs) -> Processor:
        return self._init_processor(**kwargs)

    def generate(self, i: int):
        return self.build_circuit(i=i)


class MZIPhaseFirst(AMZI):
    def __init__(self):
        super().__init__("mzi phase first")

    def build_circuit(self, **kwargs) -> Circuit:
        phi_a, phi_b, theta_a, theta_b = self._handle_params(**kwargs)
        return (Circuit(2, name="MZI")
               // (0, PS(phi=phi_a)) // BS(theta=theta_a) // (0, PS(phi=phi_b)) // BS(theta=theta_b))


class MZIPhaseLast(AMZI):
    def __init__(self):
        super().__init__("mzi phase last")

    def build_circuit(self, **kwargs) -> Circuit:
        phi_a, phi_b, theta_a, theta_b = self._handle_params(**kwargs)
        return (Circuit(2, name="MZI")
                // BS(theta=theta_a) // (1, PS(phi=phi_a)) // BS(theta=theta_b) // (1, PS(phi=phi_b)))
