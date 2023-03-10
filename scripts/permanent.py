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
import exqalibur as xq
import numpy as np
import time

m = 16
n = 12

u = pcvl.Matrix.random_unitary(m)

fsms = [[]]
fsas = [xq.FSArray(m, 0)]
coefs = [[1]]

for i in range(1, m+1):
    fsas.append(xq.FSArray(m, i))
    coefs.append(np.zeros(fsas[-1].count(), dtype=complex))
    fsms.append(xq.FSiMap(fsas[-1], fsas[-2], True))

compute = 0
def permanent(idx_current, k):
    global compute
    if k == 0:
        return 1
    if not coefs[k][idx_current]:
        m = 0
        while m < k:
            index_parent, mode = fsms[k].get(idx_current, m)
            if index_parent == xq.npos:
                break
            compute += 1
            coefs[k][idx_current] += permanent(index_parent, k-1)*u[k-1, mode]
            m += 1
    return coefs[k][idx_current]

start_slos_1 = time.time()
for idx in range(fsas[-1].count()):
    permanent(idx, n)
end_slos_1 = time.time()
time_total_slos = end_slos_1-start_slos_1
print("slos", time_total_slos)

start_qc_1 = time.time()
for idx in range(fsas[-1].count()):
    xq.permanent_cx(u, 1)
end_qc_1 = time.time()

print("qc", end_qc_1-start_qc_1)
