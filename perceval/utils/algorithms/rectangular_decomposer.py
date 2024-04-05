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

from typing import Callable, Tuple, Union

import exqalibur as xq
from perceval.components import ACircuit, Circuit, GenericInterferometer, BS, PS, catalog
from perceval.utils import Matrix, P
from perceval.serialization import deserialize_circuit


class RectangularDecomposer:

    def __init__(self, target: Union[ACircuit, Matrix], precision: float = 1e-6):
        if isinstance(target, ACircuit):
            target = target.compute_unitary()
        if target.is_symbolic():
            raise TypeError("Target must be numeric")

        self.rectangular_decomposer = xq.RectangularDecomposer(target, precision)

    def decompose(self) -> None:
        self.rectangular_decomposer.decompose()

    def get_interferometer(self, add_phase_correction: bool = False) -> Tuple[ACircuit, float]:
        return deserialize_circuit(self.rectangular_decomposer.get_interferometer(add_phase_correction))

    def get_fidelity(self) -> float:
        return self.rectangular_decomposer.get_fidelity()
