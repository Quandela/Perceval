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

r"""
This script compares building of unitary when using Circuit or directly by building matrix.
"""

import perceval as pcvl
import perceval.lib.phys as phys
import time
import numpy as np

m = 8

def phase_shift(m, theta):
    # phase shift in m x m unitary in mode 1 of angle theta
    PS = np.eye(m, dtype=complex)
    PS[0, 0] = np.cos(theta) + 1j * np.sin(theta)
    return PS

U_1 = pcvl.Matrix.random_unitary(m)
U_2 = pcvl.Matrix.random_unitary(m)

px = pcvl.P("x")
c = pcvl.Circuit(m, U_2) // (0, phys.PS(px)) // pcvl.Circuit(m, U_1)

dt_circuit = 0
dt_raw = 0

for _ in range(1000):
    top0 = time.time_ns()
    px.set_value(1)
    c.compute_unitary(use_symbolic=False)
    top1 = time.time_ns()
    dt_circuit += top1-top0

    top0 = time.time_ns()
    U = U_1 @ phase_shift(m, 1) @ U_2
    top1 = time.time_ns()
    dt_raw += top1-top0


if dt_circuit/dt_raw > 2.5:
    print("TOO_SLOW", "circuit", dt_circuit, "raw", dt_raw, "factor", dt_circuit/dt_raw)
else:
    print("OK", "circuit", dt_circuit, "raw", dt_raw, "factor", dt_circuit/dt_raw)
