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


import math
from warnings import warn

from perceval.components import ACircuit, Circuit, Port, Unitary, Processor
from perceval.components.component_catalog import CatalogItem, AsType
from perceval.utils import Encoding, PostSelect, Matrix

import numpy as np
import cmath as cm
from scipy.linalg import block_diag


def build_control_gate_unitary(n: int, alpha: float) -> Matrix:
    """ Builds the Unitary transformation of the n-qubit controlled-rotation gate C...CZ(alpha)

    :param n: number of qubits of the gate
    :param alpha: rotation angle of the gate
    :return: Unitary matrix of the post-selected gate
             Ref : https://arxiv.org/abs/2405.01395
    """

    I = Matrix.eye(n)
    a = (cm.exp(alpha * 1j) - 1) ** (1 / n)
    J = np.roll(Matrix.eye(n), shift=1, axis=1)

    # Main matrix block
    A0 = I + a * J
    B0 = Matrix.eye(n)
    M = block_diag(B0, A0)

    U0 = Matrix.get_unitary_extension(M)

    # Setting the modes right for the dual rail encoding
    initial_modes = [i for i in range(4*n)]
    final_modes = [
        2*i for i in range(n)] + [2*i+1 for i in range(n)] + [i for i in range(2*n, 4*n)]
    perm = Matrix.zeros((4*n, 4*n))
    perm[initial_modes, final_modes] = 1

    U = perm.T @ U0 @ perm
    return U


class PostProcessedControledRotationsItem(CatalogItem):
    article_ref = "https://arxiv.org/abs/2405.01395"
    description = r"""n-qubit controlled rotation gate C...CZ(alpha) with 2*n ancillary modes and a post-selection function"""
    params_doc = {
        "n": "number of qubit of the gate",
        "alpha": "angle of the controlled-rotation"
    }
    str_repr = r"""                        ╭─────╮
ctrl0 (dual rail)  ─────┤     ├───── ctrl0 (dual rail)
                   ─────┤     ├─────
                        │     │
ctrl1 (dual rail)  ─────┤     ├───── ctrl1 (dual rail)
                   ─────┤     ├─────
    .                      .           .
    .                      .           .
    .                      .           .
ctrlN (dual rail)  ─────┤     ├───── ctrlN (dual rail)
                   ─────┤     ├─────
                        ╰─────╯"""

    def __init__(self):
        super().__init__("postprocessed controlled gate")
        self._default_opts['type'] = AsType.PROCESSOR

    def build_circuit(self, **kwargs):
        """
        kwargs:
            - n : int, number of qubit of the gate.
            - alpha : float, angle of the controlled-rotation.

        :return: Circuit implementing the post-selected n-qubit controlled gate.
        """
        if not "n" in kwargs:
            raise KeyError("Missing input n")
        n = kwargs["n"]
        if not isinstance(n, int):
            raise TypeError("n must be of type int.")
        if n < 2:
            raise ValueError(f"n must be at least 2. Here n = {n}.")

        alpha = kwargs.get("alpha", math.pi)
        if not isinstance(alpha, float):
            raise TypeError("alpha must be of type float.")

        m = build_control_gate_unitary(n, alpha)
        return Circuit(4*n, name="postprocessed controlled gate").add(0, Unitary(m))

    def build_processor(self, **kwargs):
        p = self._init_processor(**kwargs)
        n = kwargs["n"]

        p.set_postselection(PostSelect('&'.join([f"[{2*n},{2*n+1}]==1" for n in range(n)])))

        for i in range(n - 1):
            p.add_port(2 * i, Port(Encoding.DUAL_RAIL, f"ctrl{i}"))
        p.add_port(2 * (n - 1), Port(Encoding.DUAL_RAIL, "data"))

        for i in range(2 * n, 4 * n):
            p.add_herald(i, 0)

        return p
