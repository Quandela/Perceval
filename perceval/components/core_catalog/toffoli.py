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
from deprecated import deprecated
from perceval.components import Circuit, Unitary, Port
from perceval.components.component_catalog import CatalogItem, AsType
from perceval.utils import Encoding, Matrix, PostSelect


class ToffoliItem(CatalogItem):
    description = r"""Toffoli gate (CCNOT) with 6 heralded modes"""
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
        super().__init__("toffoli")
        self._default_opts['type'] = AsType.PROCESSOR

    @deprecated(version="0.10.0", reason="Use build_circuit or build_processor instead")
    def build(self):
        if self._opt('type') == AsType.CIRCUIT:
            return self.build_circuit()
        elif self._opt('type') == AsType.PROCESSOR:
            return self.build_processor(backend=self._opt('backend'))

    def build_circuit(self, **kwargs):
        U = Unitary(Matrix([[3.12177488e-17 + 5.09824529e-01j, 0, 0, 0, 0, 0, 0, 0, 0, 5.26768603e-17 + 8.60278414e-01j, 0, 0],
                            [0, 5.09824529e-01, 0, 3.21169328e-01 + 5.56281593e-01j, 0, 0, 3.30393706e-01, -1.65196853e-01 - 2.86129342e-01j, -1.65196853e-01 + 2.86129342e-01j, 0, 0, 0],
                            [0, 0, -5.09824529e-01 + 6.24354977e-17j, 0, 0, 0, 0, 0, 0, 0, -8.60278414e-01 + 1.05353721e-16j, 0],
                            [0, 0, 0, 5.09824529e-01, -2.27101009e-01 - 3.93350487e-01j, 2.27101009e-01 + 3.93350487e-01j, -1.65196853e-01 + 2.86129342e-01j, 3.30393706e-01, -1.65196853e-01 - 2.86129342e-01j, 0, 0, 0],
                            [0, -2.27101009e-01 - 3.93350487e-01j, 0, 0, 5.09824529e-01, 2.36888256e-17, 1.16811815e-01 + 2.02323998e-01j, 1.16811815e-01 - 2.02323998e-01j, -2.33623630e-01, 0, 0, 6.08308700e-01],
                            [0, 2.27101009e-01 + 3.93350487e-01j, 0, 0, 2.36888256e-17, 5.09824529e-01, -1.16811815e-01 - 2.02323998e-01j, -1.16811815e-01 + 2.02323998e-01j, 2.33623630e-01, 0, 0, 6.08308700e-01],
                            [0, 3.30393706e-01, 0, -1.65196853e-01 - 2.86129342e-01j, 1.16811815e-01 - 2.02323998e-01j, -1.16811815e-01 + 2.02323998e-01j, -5.09824529e-01, 0, -3.21169328e-01 + 5.56281593e-01j, 0, 0, 0],
                            [0, -1.65196853e-01 + 2.86129342e-01j, 0, 3.30393706e-01, 1.16811815e-01 + 2.02323998e-01j, -1.16811815e-01 - 2.02323998e-01j, -3.21169328e-01 + 5.56281593e-01j, -5.09824529e-01, 0, 0, 0, 0],
                            [0, -1.65196853e-01 - 2.86129342e-01j, 0, -1.65196853e-01 + 2.86129342e-01j, -2.33623630e-01, 2.33623630e-01, 0, -3.21169328e-01 + 5.56281593e-01j, -5.09824529e-01, 0, 0, 0],
                            [8.60278414e-01, 0, 0, 0, 0, 0, 0, 0, 0, -5.09824529e-01, 0, 0],
                            [0, 0, 8.60278414e-01, 0, 0, 0, 0, 0, 0, 0, -5.09824529e-01, 0],
                            [0, 0, 0, 0, 6.08308700e-01, 6.08308700e-01, 0, 0, 0, 0, 0, -5.09824529e-01]]))
        return (Circuit(12, name="Toffoli").add(0, U))

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
