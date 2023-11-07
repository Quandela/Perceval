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

import math
from abc import ABC


from perceval.components import Processor, Circuit, BS, PS
from perceval.components.component_catalog import CatalogItem


class AMZI(CatalogItem, ABC):
    description = "Mach-Zehnder interferometer"
    params_doc = {
        'phi_a': "first phase of the MZI (default 'phi_a')",
        'phi_b': "second phase of the MZI (default 'phi_b')",
        'theta_a': "theta value of the first beam splitter (default pi/2)",
        'theta_b': "theta value of the second beam splitter (default pi/2)",
    }

    @staticmethod
    def _handle_params(**kwargs):
        if "i" in kwargs:
            kwargs["phi_a"] = f"phi_a{kwargs['i']}"
            kwargs["phi_b"] = f"phi_b{kwargs['i']}"
        return CatalogItem._handle_param(kwargs.get("phi_a", "phi_a")), \
            CatalogItem._handle_param(kwargs.get("phi_b", "phi_b")), \
            kwargs.get("theta_a", math.pi/2), \
            kwargs.get("theta_b", math.pi/2)

    def build_processor(self, **kwargs) -> Processor:
        return self._init_processor(**kwargs)

    def generate(self, i: int):
        return self.build_circuit(i=i)


_NAME_MZI_PHASE_FIRST = "mzi phase first"
_NAME_MZI_PHASE_LAST = "mzi phase last"


class MZIPhaseFirst(AMZI):
    str_repr = r"""    ╭─────╮╭─────╮╭─────╮╭─────╮
0:──┤phi_a├┤BS.Rx├┤phi_b├┤BS.Rx├──:0
    ╰─────╯│     │╰─────╯│     │
1:─────────┤     ├───────┤     ├──:1
           ╰─────╯       ╰─────╯ """
    see_also = _NAME_MZI_PHASE_LAST

    def __init__(self):
        super().__init__(_NAME_MZI_PHASE_FIRST)

    def build_circuit(self, **kwargs) -> Circuit:
        phi_a, phi_b, theta_a, theta_b = self._handle_params(**kwargs)
        return (Circuit(2, name="MZI")
               // (0, PS(phi=phi_a)) // BS(theta=theta_a) // (0, PS(phi=phi_b)) // BS(theta=theta_b))


class MZIPhaseLast(AMZI):
    str_repr = r"""    ╭─────╮       ╭─────╮
0:──┤BS.Rx├───────┤BS.Rx├─────────:0
    │     │╭─────╮│     │╭─────╮
1:──┤     ├┤phi_a├┤     ├┤phi_b├──:1
    ╰─────╯╰─────╯╰─────╯╰─────╯ """
    see_also = _NAME_MZI_PHASE_FIRST

    def __init__(self):
        super().__init__(_NAME_MZI_PHASE_LAST)

    def build_circuit(self, **kwargs) -> Circuit:
        phi_a, phi_b, theta_a, theta_b = self._handle_params(**kwargs)
        return (Circuit(2, name="MZI")
                // BS(theta=theta_a) // (1, PS(phi=phi_a)) // BS(theta=theta_b) // (1, PS(phi=phi_b)))
