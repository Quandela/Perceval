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

from perceval.components import Circuit, BS, Processor, Experiment
from perceval.components.component_catalog import CatalogItem


class Generic2ModeItem(CatalogItem):
    description = "A universal 2 mode component, implemented as a beam splitter with variable theta + 3 free phases"
    str_repr = r"""    ╭──────╮╭─────╮╭──────╮
0:──┤phi_tl├┤     ├┤phi_tr├──:0
    ╰──────╯│BS.H │╰──────╯
    ╭──────╮│theta│
1:──┤phi_bl├┤     ├──────────:1
    ╰──────╯╰─────╯ """
    params_doc = {
        "theta": "name or numerical value for beam splitter 'theta' parameter (default 'theta')",
        "phi_tl": "name or numerical value for top left phase (default 'phi_tl')",
        "phi_bl": "name or numerical value for bottom left phase (default 'phi_bl')",
        "phi_tr": "name or numerical value for top right phase (default 'phi_tr')"
    }

    def __init__(self):
        super().__init__("generic 2 mode circuit")

    def build_circuit(self, **kwargs) -> Circuit:
        return Circuit(2, name=kwargs.get("name", "U2")) \
            // BS.H(theta=self._handle_param(kwargs.get("theta", "theta")),
                    phi_tl=self._handle_param(kwargs.get("phi_tl", "phi_tl")),
                    phi_bl=self._handle_param(kwargs.get("phi_bl", "phi_bl")),
                    phi_tr=self._handle_param(kwargs.get("phi_tr", "phi_tr")))

    def build_experiment(self, **kwargs) -> Experiment:
        return Experiment(self.build_circuit(**kwargs))
