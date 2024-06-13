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
from perceval.components import Circuit, BS, Port
from perceval.components.component_catalog import CatalogItem, AsType
from perceval.components.core_catalog.heralded_cz import HeraldedCzItem
from perceval.utils import Encoding


class HeraldedCnotItem(CatalogItem):
    article_ref = ""
    description = r"""Knill CNOT gate with 2 heralded modes (built using Heralded CZ and H)."""
    str_repr = r"""                              ╭──────────╮
ctrl (dual rail) ─────────────┤          ├───────────── ctrl (dual rail)
                 ─────────────┤ Heralded ├─────────────
                      ╭───╮   │    CZ    │   ╭───╮
data (dual rail) ─────┤ H ├───┤          ├───┤ H ├───── data (dual rail)
                 ─────┤   ├───┤          ├───┤   ├─────
                      ╰───╯   ╰──────────╯   ╰───╯"""
    see_also = "heralded cz, postprocessed cnot and klm cnot"

    def __init__(self):
        super().__init__("heralded cnot")
        self._default_opts['type'] = AsType.PROCESSOR

    @deprecated(version="0.10.0", reason="Use build_circuit or build_processor instead")
    def build(self):
        if self._opt('type') == AsType.CIRCUIT:
            return self.build_circuit()
        elif self._opt('type') == AsType.PROCESSOR:
            return self.build_processor(backend=self._opt('backend'))

    def build_circuit(self, **kwargs):
        c = Circuit(6, name="Heralded CNOT")
        c.add(2, BS.H())
        heralded_cz = HeraldedCzItem()
        c.add(0, heralded_cz.build_circuit(), merge=True)
        c.add(2, BS.H())
        return c

    def build_processor(self, **kwargs):
        p = self._init_processor(**kwargs)
        return p.add_port(0, Port(Encoding.DUAL_RAIL, 'ctrl'))\
            .add_port(2, Port(Encoding.DUAL_RAIL, 'data'))\
            .add_herald(4, 1)\
            .add_herald(5, 1)
