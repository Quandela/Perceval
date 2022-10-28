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


def _post_process(s):
    return (s[1] or s[2]) and (s[3] or s[4])


class PostProcessedCnotItem(CatalogItem):
    article_ref = "https://journals.aps.org/pra/abstract/10.1103/PhysRevA.65.062324"
    description = r"""CNOT gate with 2 heralded modes and a post-processing function"""
    str_repr = r"""                      ╭─────╮
data (dual rail) ─────┤     ├───── data (dual rail)
                 ─────┤     ├─────
                      │     │
ctrl (dual rail) ─────┤     ├───── ctrl (dual rail)
                 ─────┤     ├─────
                      ╰─────╯"""

    def __init__(self):
        super().__init__("postprocessed cnot")
        self._default_opts['type'] = AsType.PROCESSOR

    def build(self):
        theta_13 = BS.r_to_theta(1/3)
        c_cnot = (Circuit(6, name="PostProcessed CNOT")
                  .add((0, 1), BS.H(theta_13, phi_bl=np.pi, phi_tr=np.pi/2, phi_tl=-np.pi/2))
                  .add((3, 4), BS.H())
                  .add((2, 3), BS.H(theta_13, phi_bl=np.pi, phi_tr=np.pi/2, phi_tl=-np.pi/2))
                  .add((4, 5), BS.H(theta_13))
                  .add((3, 4), BS.H()))

        if self._opt('type') == AsType.CIRCUIT:
            return c_cnot
        elif self._opt('type') == AsType.PROCESSOR:
            p = Processor(self._opt('backend'), c_cnot)
            p.add_herald(0, 0) \
             .add_port(1, Port(Encoding.DUAL_RAIL, 'data')) \
             .add_port(3, Port(Encoding.DUAL_RAIL, 'ctrl')) \
             .add_herald(5, 0)
            p.set_postprocess(_post_process)
            return p
