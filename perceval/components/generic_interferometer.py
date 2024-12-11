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
from collections.abc import Callable

from .linear_circuit import ACircuit, Circuit
from .unitary_components import Barrier

from perceval.utils import InterferometerShape
from perceval.utils.logging import get_logger, channel


class GenericInterferometer(Circuit):
    r"""Generate a generic interferometer circuit with generic elements and optional phase_shifter layer

    :param m: number of modes
    :param fun_gen: generator function for the building components, index is an integer allowing to generate
                    named parameters - for instance:
                    :code:`fun_gen=lambda idx: pcvl.BS()//(0, pcvl.PS(pcvl.P(f"phi_{idx}")))`
    :param shape: The output interferometer shape (InterferometerShape.RECTANGLE or InterferometerShape.TRIANGLE)
    :param depth: if None, maximal depth is :math:`m-1` for rectangular shape, :math:`m` for triangular shape.
                  Can be used with :math:`2*m` to reproduce :cite:`fldzhyan2020optimal`.
    :param phase_shifter_fun_gen: a function generating a phase_shifter circuit.
    :param phase_at_output: if True creates a layer of phase shifters at the output of the generated interferometer
                            else creates it in the input (default: False)
    :param upper_component_gen fun_gen: generator function for the building the upper component, index is an integer allowing to generate
                    named parameters - for instance:
                    :code:`fun_gen=lambda idx: pcvl.PS(pcvl.P(f"phi_upper_{idx}"))`
    :param lower_component_gen: generator function for the building the lower component, index is an integer allowing to generate
                    named parameters - for instance:
                    :code:`fun_gen=lambda idx: pcvl.PS(pcvl.P(f"phi_lower_{idx}"))`

    See :cite:`fldzhyan2020optimal`, :cite:`clements2016optimal` and :cite:`reck1994experimental`
    """
    def __init__(self,
                 m: int,
                 fun_gen: Callable[[int], ACircuit],
                 shape: InterferometerShape = InterferometerShape.RECTANGLE,
                 depth: int = None,
                 phase_shifter_fun_gen: Callable[[int], ACircuit] = None,
                 phase_at_output: bool = False,
                 upper_component_gen: Callable[[int], ACircuit] = None,
                 lower_component_gen: Callable[[int], ACircuit] = None,
                 align_with_barriers: bool = True):
        assert isinstance(shape, InterferometerShape),\
            f"Wrong type for shape, expected InterferometerShape, got {type(shape)}"
        super().__init__(m)
        self._shape = shape
        self._depth = depth
        self._depth_per_mode = [0] * m
        self._pattern_generator = fun_gen
        self._has_input_phase_layer = False
        self._upper_component_gen = upper_component_gen
        self._lower_component_gen = lower_component_gen
        self._use_barriers = align_with_barriers
        if phase_shifter_fun_gen and not phase_at_output:
            self._has_input_phase_layer = True
            for i in range(0, m):
                self.add(i, phase_shifter_fun_gen(i), merge=True)

        if shape == InterferometerShape.RECTANGLE:
            self._build_rectangle()
        elif shape == InterferometerShape.TRIANGLE:
            if upper_component_gen or lower_component_gen:
                get_logger().warn(f"upper_component_gen or lower_component_gen cannot be applied for shape {shape}")
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
    def mzi_depths(self) -> list[int]:
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

    def _add_single_mode_component(self, mode: int, component: ACircuit) -> None:
        """Add a component to the circuit, check if it's a one mode circuit

        :param mode: mode to add the component
        :param component: component to add
        """
        assert component.m == 1, f"Component should always be a one mode circuit, instead it's a {component.m} modes circuit"
        self.add(mode, component)

    def _add_upper_component(self, i_depth: int) -> None:
        """Add a component with upper_component_gen between the interferometer on the first mode

        :param i_depth: depth index of the interferometer
        """
        if self._upper_component_gen and i_depth % 2 == 1:
            self._add_single_mode_component(0, self._upper_component_gen(i_depth // 2))

    def _add_lower_component(self, i_depth: int) -> None:
        """Add a component with lower_component_gen between the interferometer on the last mode

        :param i_depth: depth index of the interferometer
        """
        # If m is even, the component is added at even depth index, else it's added in at odd depth index
        if (self._lower_component_gen and
                ((i_depth % 2 == 1 and self.m % 2 == 0) or (i_depth % 2 == 0 and self.m % 2 == 1))):
            self._add_single_mode_component(self.m - 1, self._lower_component_gen(i_depth // 2))

    def _build_rectangle(self):
        max_depth = self.m if self._depth is None else self._depth
        idx = 0
        for i in range(0, max_depth):
            self._add_upper_component(i)
            for j in range(0+i%2, self.m-1, 2):
                if self._depth is not None and (self._depth_per_mode[j] == self._depth
                                                or self._depth_per_mode[j+1] == self._depth):
                    continue
                self.add((j, j+1), self._pattern_generator(idx), merge=True)
                self._depth_per_mode[j] += 1
                self._depth_per_mode[j+1] += 1
                idx += 1
            self._add_lower_component(i)
            if self._use_barriers:
                self.add(0, Barrier(self.m, visible=False))

    def _build_triangle(self):
        idx = 0
        for i in range(1, self.m):
            for j in range(i, 0, -1):
                if self._depth is not None and (self._depth_per_mode[j-1] == self._depth
                                                or self._depth_per_mode[j] == self._depth):
                    continue
                self.add((j-1, j), self._pattern_generator(idx), merge=True)
                if self._use_barriers:
                    self.add((j-1, j), Barrier(2, visible=False))
                self._depth_per_mode[j-1] += 1
                self._depth_per_mode[j] += 1
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

    def set_param_list(self, param_list: list[float], top_left_pos: tuple[int, int], m: int):
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
            get_logger().warn("set_param_list was designed for rectangular interferometer", channel.user)
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

    def set_params_from_other(self, other: Circuit, top_left_pos: tuple[int, int]):
        """Retrieve parameter value from another interferometer

        :param other: Another interferometer
        :param top_left_pos: Starting position of the insertion. See full description in `set_param_list`
        """
        if not other.defined:
            raise ValueError("Cannot copy parameters from a circuit which isn't fully defined")
        param_list = [float(p) for p in other.get_parameters()]
        self.set_param_list(param_list, top_left_pos, other.m)
