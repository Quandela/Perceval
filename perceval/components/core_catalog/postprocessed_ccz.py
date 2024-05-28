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

from math import pi

from perceval.components import Circuit, Port, Unitary
from perceval.components.component_catalog import CatalogItem, AsType
from perceval.components.core_catalog import controlled_rotation_gates
from perceval.utils import Encoding, PostSelect, Matrix


class PostProcessedCCZItem(CatalogItem):
    article_ref = "https://journals.aps.org/pra/abstract/10.1103/PhysRevA.65.062324"
    description = r"""CCZ gate with 6 heralded modes and a post-selection function"""
    str_repr = r"""                       ╭─────╮
ctrl0 (dual rail) ─────┤     ├───── ctrl0 (dual rail)
                  ─────┤     ├─────
                       │     │
ctrl1 (dual rail) ─────┤     ├───── ctrl1 (dual rail)
                  ─────┤     ├─────
                       │     │
data (dual rail)  ─────┤     ├───── data (dual rail)
                  ─────┤     ├─────
                       ╰─────╯"""

    def __init__(self):
        super().__init__("postprocessed ccz")
        self._default_opts['type'] = AsType.PROCESSOR

    def build_circuit(self, **kwargs):
        m = Matrix(controlled_rotation_gates.build_control_gate_unitary(3, pi))
        return Circuit(12, name="PostProcessed CCZ").add(0, Unitary(m))

    def build_processor(self, **kwargs):
        p = self._init_processor(**kwargs)
        p.set_postselection(PostSelect("[0,1]==1 & [2,3]==1 & [4,5]==1"))
        return p.add_port(0, Port(Encoding.DUAL_RAIL, 'ctrl0')) \
            .add_port(2, Port(Encoding.DUAL_RAIL, 'ctrl1')) \
            .add_port(4, Port(Encoding.DUAL_RAIL, 'data')) \
            .add_herald(6, 0) \
            .add_herald(7, 0) \
            .add_herald(8, 0) \
            .add_herald(9, 0) \
            .add_herald(10, 0) \
            .add_herald(11, 0)
