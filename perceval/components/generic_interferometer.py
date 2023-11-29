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
from typing import Callable, List, Optional, Tuple

from .linear_circuit import ACircuit, Circuit
from perceval.utils import InterferometerShape


class GenericInterferometer(Circuit):
    r"""Generate a generic interferometer circuit with generic elements and optional phase_shifter layer

    :param m: number of modes
    :param fun_gen: generator function for the building components, index is an integer allowing to generate
                    named parameters - for instance:
                    :code:`fun_gen=lambda idx: phys.BS()//(0, phys.PS(pcvl.P(f"phi_{idx}")))`
    :param shape: The output interferometer shape (InterferometerShape.RECTANGLE or InterferometerShape.TRIANGLE)
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
                 shape: InterferometerShape = InterferometerShape.RECTANGLE,
                 depth: int = None,
                 phase_shifter_fun_gen: Optional[Callable[[int], ACircuit]] = None,
                 phase_at_output: bool = False):
        assert isinstance(shape, InterferometerShape),\
            f"Wrong type for shape, expected InterferometerShape, got {type(shape)}"
        super().__init__(m)
        self._shape = shape
        self._depth = depth
        self._depth_per_mode = [0] * m
        self._pattern_generator = fun_gen
        self._has_input_phase_layer = False
        if phase_shifter_fun_gen and not phase_at_output:
            self._has_input_phase_layer = True
            for i in range(0, m):
                self.add(i, phase_shifter_fun_gen(i), merge=True)

        if shape == InterferometerShape.RECTANGLE:
            self._build_rectangle()
        elif shape == InterferometerShape.TRIANGLE:
            self._build_triangle()
        else:
            raise NotImplementedError(f"Shape {shape} not supported")

        self._has_output_phase_layer = False
        if phase_shifter_fun_gen and phase_at_output:
            self._has_output_phase_layer = True
            for i in range(0, m):
                self.add(i, phase_shifter_fun_gen(i))

    def __repr__(self):
        return f"Generic interferometer ({self.m} modes, {str(self._shape.name)}, {self.ncomponents()} components)"

    @property
    def mzi_depths(self) -> List[int]:
        """Return a list of MZI depth, per mode"""
        return self._depth_per_mode

    def remove_phase_layer(self):
        """Remove the optional phase layer at the input or at the output.
        Does nothing if such a layer does not exist"""
        if self._has_input_phase_layer:
            self._components = self._components[self.m:]
        if self._has_output_phase_layer:
            self._components = self._components[:-self.m]

    def set_identity_mode(self):
        """Set the interferometer in identity mode (i.e. photons are output on the mode they're input)"""
        for p in self.get_parameters():
            p.set_value(math.pi)

    def _build_rectangle(self):
        max_depth = self.m if self._depth is None else self._depth
        idx = 0
        for i in range(0, max_depth):
            for j in range(0+i%2, self.m-1, 2):
                if self._depth is not None and (self._depth_per_mode[j] == self._depth
                                                or self._depth_per_mode[j+1] == self._depth):
                    continue
                self.add((j, j+1), self._pattern_generator(idx), merge=True, x_grid=i)
                self._depth_per_mode[j] += 1
                self._depth_per_mode[j+1] += 1
                idx += 1

    def _build_triangle(self):
        idx = 0
        # record the physical position of each block for aligning purpose
        pos_mode = [0] * self.m
        for i in range(1, self.m):
            for j in range(i, 0, -1):
                if self._depth is not None and (self._depth_per_mode[j-1] == self._depth
                                                or self._depth_per_mode[j] == self._depth):
                    continue
                pos_component = max(pos_mode[j-1], pos_mode[j])
                self.add((j-1, j), self._pattern_generator(idx), merge=True, x_grid=pos_component)
                self._depth_per_mode[j-1] += 1
                self._depth_per_mode[j] += 1
                pos_mode[j-1] = pos_component+1
                pos_mode[j] = pos_component+1
                idx += 1

    def _find_param_index(self, col: int, lin: int, even_col_size: int, odd_col_size: int) -> int:
        p_idx = (even_col_size + odd_col_size) * (col // 2)
        if col % 2 == 1:
            p_idx += even_col_size
        p_idx += lin * 2
        if self._has_input_phase_layer:
            p_idx += self.m
        return p_idx

    @staticmethod
    def _compute_insertion_depth(start_col: int, m: int, param_count: int) -> int:
        depth = 0
        cc = 0
        k = 0
        while k < param_count:
            depth += 1
            k += 2*(m//2) if (start_col+cc)%2 == 0 else 2*((m-1)//2)
            cc += 1
        return depth

    def set_param_list(self, param_list: List[float], top_left_pos: Tuple[int, int], m: int):
        """Insert parameters value starting from a given position in the interferometer.

        This method is designed to work on rectangular interferometers

        :param param_list: List of numerical values for the parameters
        :param top_left_pos: Starting position of the insertion (column#, row#). Position is handled MZI-wise (i.e.
        (0,0) starts inserting values on the top-left-most MZI of the interferometer whereas (1,0) starts on top of the
        2nd MZI column). The optional phase layer is ignored in the position handling.
        :param m: Mode count on where to insert the parameter values
        """
        col, lin = top_left_pos
        depth = self._compute_insertion_depth(col, m, len(param_list))
        if col < 0 or col+depth-1 >= self.m:
            raise ValueError(f"Invalid param list width, expected interval in [0,{self.m}], got [{col},{col+depth-1}]")
        if lin < 0 or lin+m >= self.m:
            raise ValueError(f"Invalid param list height, expected interval in [0,{self.m}], got [{lin},{lin+m}]")
        if self._shape != InterferometerShape.RECTANGLE:
            warn("set_param_list was designed for rectangular interferometer")

        even_mode_count = self.m % 2 == 0
        even_col_size = self.m - (self.m % 2)
        odd_col_size = even_col_size - 2 if even_mode_count else even_col_size
        self_params = self.get_parameters()
        cc = 0  # current col
        k = 0
        while k < len(param_list):
            start = self._find_param_index(col+cc, lin, even_col_size, odd_col_size)
            param_count = 2*(m//2) if (col+cc) % 2 == 0 else 2*((m-1)//2)
            for j in range(param_count):
                self_params[start+j].set_value(param_list[k])
                k += 1
                if k == len(param_list):
                    break
            cc += 1

    def set_params_from_other(self, other: Circuit, top_left_pos: Tuple[int, int]):
        """Retrieve parameter value from another interferometer

        :param other: Another interferometer
        :param top_left_pos: Starting position of the insertion. See full description in `set_param_list`
        """
        if not other.defined:
            raise ValueError("Cannot copy parameters from a circuit which isn't fully defined")
        param_list = [float(p) for p in other.get_parameters()]
        self.set_param_list(param_list, top_left_pos, other.m)
