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

import numpy as np

from perceval.utils.matrix import Matrix


def fidelity(u: Matrix, v: Matrix) -> float:
    r""" Calculate the fidelity of a unitary implementation compared to a reference unitary

    :param u: the unitary to evaluate
    :param v: the reference unitary
    :return: real [0-1] float fidelity
    """
    u_dag = np.transpose(np.conjugate(u))
    f = abs(np.trace(u_dag @ v)) ** 2 / (u.shape[0] * np.trace(u_dag @ u))
    if isinstance(f, complex):
        return f.real
    else:
        return f

def modulus_fidelity(u: Matrix, v: Matrix) -> float:
    r""" Calculate the fidelity of a unitary implementation compared to a reference unitary just comparing
    single input-single output probabilities

    :param u: the unitary to evaluate
    :param v: the reference unitary
    :return: real [0-1] float fidelity
    """
    u_dag = np.transpose(np.conjugate(u))
    f = np.trace(np.dot(abs(u_dag), abs(v)))/u.shape[0]
    if isinstance(f, complex):
        return f.real
    else:
        return f

def frobenius(u: Matrix, v: Matrix) -> float:
    r""" Frobenius norm

    :param u: the unitary to evaluate
    :param v: the reference unitary
    :return: real [0-1] float fidelity
    """
    return np.linalg.norm(u - v)
