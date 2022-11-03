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

"""
This script checks overhead when using circuit objects compared to direct building of unitary matrix
"""

import perceval as pcvl
import perceval.components.unitary_components as comp
import numpy as np
import time
from collections import defaultdict

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--m', default=5, type=int, help='number of photons')
parser.add_argument('--iter', default=1000, type=int, help='number of iterations')
args = parser.parse_args()

mtime = defaultdict(float)

for _ in range(args.iter):
    U_1 = pcvl.Matrix.random_unitary(args.m)
    U_2 = pcvl.Matrix.random_unitary(args.m)
    px = pcvl.P("x")
    top0 = time.time_ns()
    c = comp.Unitary(U_2) // (0, comp.PS(px)) // comp.Unitary(U_1)

    top1 = time.time_ns()
    px.set_value(0.5)
    U = c.compute_unitary(use_symbolic=False)
    top2 = time.time_ns()

    def phase_shift(m, theta):
        # phase shift in m x m unitary in mode 1 of angle theta
        PS = np.eye(m, dtype=complex)
        PS[0, 0] = np.cos(theta) + 1j * np.sin(theta)
        return PS
    phase_shift_0 = phase_shift(args.m, 0.5)
    U_0 = np.matmul(U_2, phase_shift_0)
    U_0 = np.matmul(U_0, U_1)
    top3 = time.time_ns()

    mtime["build"] += top1-top0
    mtime["compute"] += top2-top1
    mtime["manual"] += top3-top2

for k, v in mtime.items():
    print(k, v)
