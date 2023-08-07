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

from .linear_circuit import ACircuit, Circuit

from typing import Callable, Optional


class GenericInterferometer(Circuit):
    r"""Generate a generic interferometer circuit with generic elements and optional phase_shifter layer

    :param m: number of modes
    :param fun_gen: generator function for the building components, index is an integer allowing to generate
                    named parameters - for instance:
                    :code:`fun_gen=lambda idx: phys.BS()//(0, phys.PS(pcvl.P("phi_%d"%idx)))`
    :param shape: `rectangle` or `triangle`
    :param depth: if None, maximal depth is :math:`m-1` for rectangular shape, :math:`m` for triangular shape.
                  Can be used with :math:`2*m` to reproduce :cite:`fldzhyan2020optimal`.
    :param phase_shifter_fun_gen: a function generating a phase_shifter circuit.
    :param phase_at_output: if True creates a layer of phase shifters at the output of the generated interferometer
                            else creates it in the input (default: False)

    See :cite:`fldzhyan2020optimal`, :cite:`clements2016optimal` and :cite:`reck1994experimental`
    """
    def __init__(self,
                 m: int,
                 fun_gen: Callable[[int], ACircuit],
                 shape: str = "rectangle",  # Literal["triangle", "rectangle"]
                 depth: int = None,
                 phase_shifter_fun_gen: Optional[Callable[[int], ACircuit]] = None,
                 phase_at_output: bool = False):
        super().__init__(m)
        self._depth = depth
        self._depth_per_mode = [0] * m
        self._pattern_generator = fun_gen

        if phase_shifter_fun_gen and not phase_at_output:
            for i in range(0, m):
                self.add(i, phase_shifter_fun_gen(i), merge=True)

        if shape == "rectangle":
            self._build_rectangle()
        elif shape == "triangle":
            self._build_triangle()
        else:
            raise ValueError(f"Supported shapes are 'triangle' and 'recangle' (got {shape} instead)")

        if phase_shifter_fun_gen and phase_at_output:
            for i in range(0, m):
                self.add(i, phase_shifter_fun_gen(i))

    @property
    def depths(self):
        return self._depth_per_mode

    def _build_rectangle(self):
        max_depth = self.m if self._depth is None else self._depth
        idx = 0
        for i in range(0, max_depth):
            for j in range(0+i%2, self.m-1, 2):
                if self._depth is not None and (self._depth_per_mode[j] == self._depth
                                                or self._depth_per_mode[j+1] == self._depth):
                    continue
                self.add((j, j+1), self._pattern_generator(idx), merge=True)
                self._depth_per_mode[j] += 1
                self._depth_per_mode[j+1] += 1
                idx += 1

    def _build_triangle(self):
        idx = 0
        for i in range(1, self.m):
            for j in range(i, 0, -1):
                if self._depth is not None and (self._depth_per_mode[j-1] == self._depth
                                                or self._depth_per_mode[j] == self._depth):
                    continue
                self.add((j-1, j), self._pattern_generator(idx), merge=True)
                self._depth_per_mode[j-1] += 1
                self._depth_per_mode[j] += 1
                idx += 1
