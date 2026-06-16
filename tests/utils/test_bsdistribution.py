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

import copy

from exqalibur import FSDistribution, FockState, FSFunction
from perceval import BSDistribution
from math import sqrt
import pytest

num_i = 12
num_f = 3.14159
num_c = complex(0.5, 0.5*sqrt(3))
fs = FockState()
bsd = BSDistribution(FSDistribution(fs))
bsf = BSDistribution(FSFunction())

def test_constructors():
    BSDistribution()
    BSDistribution(fs)
    BSDistribution(bsd)
    BSDistribution(bsf)
    BSDistribution({fs: 1.0})

    with pytest.raises(Exception):
        BSDistribution({FockState([1,0]): 0.5, FockState([1]): 0.5})
    with pytest.raises(Exception):
        BSDistribution({fs: num_c})

def test_operators():
    bsd *  fs
    fs  *  bsd
    bsd *  bsd
    bsf *  fs
    fs  *  bsf
    bsf *  bsf
    bsf *  bsd
    bsd *  bsf
    bsf == bsf
    bsf != bsf
    bsd == bsf
    bsd != bsf
    bsf == bsd
    bsf != bsd
    bsd == bsd
    bsd != bsd

    - bsd
    - bsf

    bsd   * num_i
    bsf   * num_i
    num_i * bsd
    num_i * bsf
    bsd   / num_i
    bsf   / num_i

    bsd   * num_f
    bsf   * num_f
    num_f * bsd
    num_f * bsf
    bsd   / num_f
    bsf   / num_f

    bsf ** num_i
    with pytest.raises(Exception):
        bsf ** num_f
    with pytest.raises(Exception):
        bsf ** num_c

    bsdd = copy.copy(bsd)
    bsdd += bsd
    bsdd = copy.copy(bsd)
    bsdd += bsf
    bsdd = copy.copy(bsd)
    bsdd -= bsd
    bsdd = copy.copy(bsd)
    bsdd -= bsf
    bsdd = copy.copy(bsd)
    bsdd *= num_f
    bsdd = copy.copy(bsd)
    bsdd *= fs
    bsdd = copy.copy(bsd)
    bsdd *= bsd
    bsdd = copy.copy(bsd)
    bsdd *= bsf
    bsdd = copy.copy(bsd)
    bsdd /= num_f

    bsff = copy.copy(bsf)
    bsff += bsd
    bsff = copy.copy(bsf)
    bsff += bsf
    bsff = copy.copy(bsf)
    bsff -= bsd
    bsff = copy.copy(bsf)
    bsff -= bsf
    bsff = copy.copy(bsf)
    bsff *= num_f
    bsff = copy.copy(bsf)
    bsff *= fs
    bsff = copy.copy(bsf)
    bsff *= bsd
    bsff = copy.copy(bsf)
    bsff *= bsf
    bsff = copy.copy(bsf)
    bsff /= num_f

    with pytest.raises(Exception):
        bsf   * num_c
    with pytest.raises(Exception):
        num_c * bsf
    with pytest.raises(Exception):
        bsf   / num_c
