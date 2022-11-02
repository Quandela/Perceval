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

import perceval as pcvl
import numpy as np
import matplotlib.pyplot as plt
import perceval.components.unitary_components as comp
from perceval.rendering.pdisplay import pdisplay_statevector


N = 100
ind = np.arange(0, 1, 1/N)
o = np.zeros((100,))
simulator_backend = pcvl.BackendFactory().get_backend('Naive')

for i in range(N):
    source = pcvl.Source(brightness=1, purity=1, indistinguishability=ind[i])
    qpu = pcvl.Processor({0: source, 1: source}, comp.BS())
    all_p, sv_out = qpu.run(simulator_backend)
    o[i] = sv_out[pcvl.StateVector("|1,1>")]

plt.plot(ind, o)
plt.ylabel("$p(|1,1>)$")
plt.xlabel("indistinguishability")
source = pcvl.Source(brightness=1, purity=1, indistinguishability=0.5)
qpu = pcvl.Processor({0: source, 1: source}, comp.BS())

all_p, sv_out = qpu.run(simulator_backend)
print("INPUT\n", pdisplay_statevector(qpu.source_distribution))
print("OUTPUT\n", pdisplay_statevector(sv_out))

plt.show()
