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
import quandelibc as qc
import thewalrus
import numpy as np
import time

# benchmark inspired from https://the-walrus.readthedocs.io/en/latest/gallery/permanent_tutorial.html

a0 = 300.
anm1 = 2
n = 28
r = (anm1/a0)**(1./(n-1))
nreps = [(int)(a0*(r**((i)))) for i in range(n)]

times_walrus = np.empty(n)
times_qc_1 = np.empty(n)
times_qc_4 = np.empty(n)
times_qc_0 = np.empty(n)


for ind, reps in enumerate(nreps):
    #print(ind+1,reps)
    matrices = []
    for i in range(reps):
        size = ind+1
        nth = 1
        matrices.append(pcvl.Matrix.random_unitary(size))
    start_walrus = time.time()
    for matrix in matrices:
        res = thewalrus.perm(matrix)
    end_walrus = time.time()
    start_qc_1 = time.time()
    for matrix in matrices:
        res = qc.permanent_cx(matrix, 2)
    end_qc_1 = time.time()
    start_qc_4 = time.time()
    for matrix in matrices:
        res = qc.permanent_cx(matrix, 4)
    end_qc_4 = time.time()
    start_qc_0 = time.time()
    for matrix in matrices:
        res = qc.permanent_cx(matrix, 0)
    end_qc_0 = time.time()


    times_walrus[ind] = (end_walrus - start_walrus)/reps
    times_qc_1[ind] = (end_qc_1 - start_qc_1)/reps
    times_qc_4[ind] = (end_qc_4 - start_qc_4)/reps
    times_qc_0[ind] = (end_qc_0 - start_qc_0)/reps



    print(ind+1, times_walrus[ind], times_qc_1[ind], times_qc_4[ind], times_qc_0[ind])

import matplotlib.pyplot as plt
# %matplotlib inline
# %config InlineBackend.figure_formats=['svg']
plt.semilogy(np.arange(1,n+1),times_walrus,"+")
plt.semilogy(np.arange(1,n+1),times_qc_1,"*")
plt.semilogy(np.arange(1,n+1),times_qc_4,"-")
plt.semilogy(np.arange(1,n+1),times_qc_0,"x")

plt.xlabel(r"Matrix size $n$")
