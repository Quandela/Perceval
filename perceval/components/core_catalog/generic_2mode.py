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

from perceval.utils.parameter import P
from perceval.components import Circuit, Processor
from perceval.components.base_components import *
from perceval.components.component_catalog import CatalogItem, AsType


class Generic2ModeItem(CatalogItem):
    def __init__(self):
        super().__init__("generic 2 mode circuit")
        self._default_opts['type'] = AsType.CIRCUIT

    def build(self):
        c = Circuit(2) // GenericBS(theta=P("theta"), phi_a=P("phi_a"), phi_b=P("phi_b"), phi_d=P("phi_d"))
        if self._opt('type') == AsType.CIRCUIT:
            return c
        elif self._opt('type') == AsType.PROCESSOR:
            p = Processor()
            return p.add(0, c)



# With simple BS convention:
# c = SimpleBS(theta=P("theta"), phi=P("phi")) // PS(phi=P("phi_a")) // (1, PS(phi=P("phi_b")))

# generic_2mode = PredefinedCircuit(c, "generic 2 mode circuit")
