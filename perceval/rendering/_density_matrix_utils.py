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

from matplotlib import ticker
from matplotlib import colormaps
from cmath import phase, pi
import numpy as np
from numpy.linalg import norm
import math


def _complex_to_rgb(z: complex, cmap='hsv'):
    """for better rendering, cmap should be a cyclic matplotlib ColorMap"""
    r, g, b, a = colormaps[cmap]((phase(z) + pi) / (2 * pi))
    a = abs(z)
    vect = np.array([r, g, b])
    vect = (a / norm(vect)) * vect
    return vect


def _csr_to_rgb(matrix, cmap='hsv'):
    """convert a complex csr_matrix to an rgb image"""
    if matrix.ndim != 2:
        raise ValueError(f"matrix should be a 2d array, not {matrix.ndim}d")

    img = np.zeros(matrix.shape + (3,))
    coef_max = 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[0]):
            z = matrix[i, j]
            if z != 0:
                img[i, j, :] = _complex_to_rgb(z, cmap)
                if abs(z) > coef_max:
                    coef_max = abs(z)
    img = (1/coef_max) * img
    return img


def _csr_to_greyscale(matrix):
    """convert a complex matrix to a greyscale image"""
    if matrix.ndim != 2:
        raise ValueError(f"matrix should be a 2d array, not {matrix.ndim}d")

    img = np.zeros(matrix.shape)
    coef_max = 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[0]):
            z = matrix[i, j]
            if z != 0:
                img[i, j] = 255*abs(z)
                if abs(z) > coef_max:
                    coef_max = abs(z)
    img = (1 / coef_max) * img
    return img


def generate_ticks(dm):
    m, n = dm.m, dm.n_max
    tick_list = [0]
    tick_labels = ["0 photon"]
    for k in range(n):
        tick_list.append(math.comb(m+k, k))
        tick_labels.append(str(k+1)+" photons")
    return tick_list, tick_labels
