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

import numpy as np

from perceval.components import PredefinedCircuit
import perceval.lib.phys as phys

c_cnot = (phys.Circuit(6, name="PostProcessed CNOT")
              .add((0, 1), phys.BS(R=1 / 3, phi_b=np.pi, phi_d=0))
              .add((3, 4), phys.BS(R=1 / 2))
              .add((2, 3), phys.BS(R=1 / 3, phi_b=np.pi, phi_d=0))
              .add((4, 5), phys.BS(R=1 / 3))
              .add((3, 4), phys.BS(R=1 / 2)))


def _post_process(s):
    return (s[1] or s[2]) and (s[3] or s[4])


postprocessed_cnot = PredefinedCircuit(c_cnot,
                                       "postprocessed cnot",
                                       description="https://journals.aps.org/pra/abstract/10.1103/PhysRevA.65.062324",
                                       heralds={0: 0, 5: 0},
                                       post_select_fn=_post_process)
