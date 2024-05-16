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

import math as m

from deprecated import deprecated
from perceval.components import Circuit, Processor
from perceval.components.unitary_components import *
from perceval.components.component_catalog import CatalogItem, AsType
from perceval.components.port import Port, Encoding


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
    
    def __init__(self):
        super().__init__("heralded cz")
        self._default_opts['type'] = AsType.PROCESSOR

    @deprecated(version="0.10.0", reason="Use build_circuit or build_processor instead")
    def build(self):
        if self._opt('type') == AsType.CIRCUIT:
            return self.build_circuit()
        elif self._opt('type') == AsType.PROCESSOR:
            return self.build_processor(backend=self._opt('backend'))

    def build_circuit(self, **kwargs) -> Circuit:
        U = Matrix(
            np.array(
                [[1,0,0,0,0,0],
                 [0,1,0,0,0,0],
                 [0,0,-1/3,-m.sqrt(2)/3,m.sqrt(2)/3,2/3],
                 [0,0,m.sqrt(2)/3,-1/3,-2/3,m.sqrt(2)/3],
                 [0,0,-m.sqrt(3+m.sqrt(6))/3,m.sqrt(3-m.sqrt(6))/3,-m.sqrt((3+m.sqrt(6))/2)/3,m.sqrt(1/6-1/(3*m.sqrt(6)))],
                 [0,0,-m.sqrt(3-m.sqrt(6))/3,-m.sqrt(3+m.sqrt(6))/3,-m.sqrt(1/6-1/(3*m.sqrt(6))),-m.sqrt((3+m.sqrt(6))/2)/3]]))



        return (Circuit(6, name="Heralded CZ")
                .add(0, PERM([0,2,1,3]))
                .add(0, Unitary(U), merge=True)
                .add(0, PERM([0,2,1,3])))

    def build_processor(self, **kwargs) -> Processor:
        p = self._init_processor(**kwargs)
        return p.add_port(0, Port(Encoding.DUAL_RAIL, 'ctrl')) \
            .add_port(2, Port(Encoding.DUAL_RAIL, 'data')) \
            .add_herald(4, 1) \
            .add_herald(5, 1)
