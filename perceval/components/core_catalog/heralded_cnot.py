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

from perceval.components import Circuit, Processor, BS, PERM, Port
from perceval.components.component_catalog import CatalogItem, AsType
from perceval.utils import Encoding


class HeraldedCnotItem(CatalogItem):
    article_ref = "https://doi.org/10.1073/pnas.1018839108"
    description = r"""CNOT gate with 4 heralded modes"""
    str_repr = r"""                      ╭─────╮
ctrl (dual rail) ─────┤     ├───── ctrl (dual rail)
                 ─────┤     ├─────
                      │     │
data (dual rail) ─────┤     ├───── data (dual rail)
                 ─────┤     ├─────
                      ╰─────╯"""

    R1 = 0.228
    R2 = 0.758
    theta1 = BS.r_to_theta(R1)
    theta2 = BS.r_to_theta(R2)

    def __init__(self):
        super().__init__("heralded cnot")
        self._default_opts['type'] = AsType.PROCESSOR

    def build(self):
        c_hcnot = (Circuit(8, name="Heralded CNOT")
                   .add(1, PERM([2, 4, 3, 0, 1]))
                   .add(4, BS.H())
                   .add(3, PERM([1, 3, 0, 4, 2]))
                   .add(3, BS.H())
                   .add(3, PERM([2, 0, 1]))
                   .add(2, BS.H(theta=self.theta1))
                   .add(4, BS.H(theta=self.theta1))
                   .add(3, PERM([1, 2, 0]))
                   .add(3, BS.H())
                   .add(1, PERM([2, 0, 3, 1, 6, 5, 4]))
                   .add(2, BS.H(theta=self.theta2))
                   .add(2, PERM([1, 0]))
                   .add(4, BS.H(theta=self.theta2))
                   .add(4, PERM([1, 2, 0]))
                   .add(4, BS.H())
                   .add(1, PERM([4, 3, 0, 2, 1])))

        if self._opt('type') == AsType.CIRCUIT:
            return c_hcnot
        elif self._opt('type') == AsType.PROCESSOR:
            p = Processor(self._opt('backend'), c_hcnot)
            return p.add_port(0, Port(Encoding.DUAL_RAIL, 'ctrl')) \
                    .add_port(2, Port(Encoding.DUAL_RAIL, 'data')) \
                    .add_herald(4, 0) \
                    .add_herald(5, 1) \
                    .add_herald(6, 0) \
                    .add_herald(7, 1)
