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

from perceval.components import Circuit, Processor
from perceval.components.unitary_components import *
from perceval.components.component_catalog import CatalogItem, AsType
from perceval.components.port import Herald, Port, Encoding


class HeraldedCzItem(CatalogItem):
    article_ref = "https://arxiv.org/abs/quant-ph/0110144"
    description = r"""CZ gate with 2 heralded modes"""
    str_repr = r"""                      ╭─────╮
ctrl (dual rail) ─────┤     ├───── ctrl (dual rail)
                 ─────┤     ├─────
                      │     │
data (dual rail) ─────┤     ├───── data (dual rail)
                 ─────┤     ├─────
                      ╰─────╯"""

    theta1 = 2*np.pi*54.74/180
    theta2 = 2*np.pi*17.63/180
    #The additional 2 factor takes into account the difference between the beam-splitter conventions of Perceval and the paper

    def __init__(self):
        super().__init__("heralded cz")
        self._default_opts['type'] = AsType.PROCESSOR

    def build(self):
        #the matrix of this first circuit is the same as the one presented in the reference paper, the difference in the second phase shift - placed on mode 3 instead of mode 1 - is due to a different convention for the beam-splitters (signs inverted in second column).
        last_modes_cz = (Circuit(4)
            .add(0, PS(np.pi))
            .add(3, PS(np.pi))
            .add((1, 2), PERM([1, 0]))
            .add((0, 1), BS.H(theta=self.theta1))
            .add((2, 3), BS.H(theta=self.theta1))
            .add((1, 2), PERM([1, 0]))
            .add((0, 1), BS.H(theta=-self.theta1))
            .add((2, 3), BS.H(theta=self.theta2)))

        c_hcz=(Circuit(6, name="Heralded CZ")
            .add((1, 2), PERM([1, 0]))
            .add((2, 3, 4, 5), last_modes_cz, merge=True)
            .add((1, 2), PERM([1, 0])))

        if self._opt('type') == AsType.CIRCUIT:
            return c_hcz
        elif self._opt('type') == AsType.PROCESSOR:
            p = Processor(self._opt('backend'), c_hcz)
            return p.add_port(0, Port(Encoding.DUAL_RAIL, 'ctrl')) \
                    .add_port(2, Port(Encoding.DUAL_RAIL, 'data')) \
                    .add_herald(4, 1) \
                    .add_herald(5, 1)
