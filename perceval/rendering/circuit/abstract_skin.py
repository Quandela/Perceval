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

from abc import ABC, abstractmethod
from multipledispatch import dispatch

from perceval.components import ACircuit, AFFConfigurator, AProcessor, PERM, AComponent, TD, LC
from perceval.utils import format_parameters, ModeType


class ASkin(ABC):
    """
    Abstract skin
    -------------
    A skin is required in the use of pdisplay for the following formats:
    - Format.HTML
    - Format.MPLOT

    A skin has three major responsibilities:
    - measuring the display size of a component / composite circuit
    - providing shape functions to draw individual components
    - exposing style data (stroke style, colors, etc.)
    """

    def __init__(self, photonic_style: dict, style_subcircuit: dict, compact_display: bool = False):
        self._compact = compact_display
        self.style = {ModeType.PHOTONIC: photonic_style,
                      ModeType.HERALD: {"stroke": None, "stroke_width": 1},
                      ModeType.CLASSICAL: {"stroke": "blue", "stroke_width": 1}
                      }
        self.style_subcircuit = style_subcircuit
        self.precision: float = 1e-6
        self.nsimplify: bool = True

    @dispatch((ACircuit, TD, LC), bool)
    def get_size(self, c: ACircuit, recursive: bool = False) -> tuple[int, int]:
        """Gets the size of a circuit in arbitrary unit. If composite, it will take its components into account"""
        if not c.is_composite():
            return self.measure(c)

        # w represents the graph of the circuit.
        # Each value being the output of the rightmost component on the corresponding mode
        w = [0] * c.m
        for modes, comp in c._components:
            r = slice(modes[0], modes[0]+comp.m)
            start_w = max(w[r])
            if comp.is_composite() and recursive:
                comp_width, _ = self.get_size(comp, False)
            else:
                comp_width = self.get_width(comp)
            end_w = start_w + comp_width
            w[r] = [end_w] * comp.m
        return max(w), c.m

    @dispatch(AProcessor, bool)
    def get_size(self, p: AProcessor, recursive: bool = False) -> tuple[int, int]:
        height = p.m
        # w represents the graph of the circuit.
        # Each value being the output of the rightmost component on the corresponding mode
        w = [0] * p.circuit_size
        for modes, comp in p._components:
            if not isinstance(comp, PERM):
                height = max(height, comp.m + modes[0])

            r = slice(modes[0], modes[0] + comp.m)
            start_w = max(w[r])
            if comp.is_composite() and recursive:
                comp_width, _ = self.get_size(comp, False)
            elif isinstance(comp, AFFConfigurator) and recursive:
                comp_width, _ = self.get_size(comp.circuit_template(), False)
            else:
                comp_width = self.get_width(comp)
            end_w = start_w + comp_width
            w[r] = [end_w] * comp.m
        return max(w) + 1, min(p.circuit_size, height+2)

    def measure(self, c: AComponent) -> tuple[int, int]:
        """
        Returns the measure (in arbitrary unit (AU) where the space between two modes = 1 AU)
        of a single component treated as a block (meaning that a composite circuit will not be
        measured recursively. Use get_size() instead)
        """
        return self.get_width(c), c.m

    @abstractmethod
    def get_width(self, c) -> int:
        """Returns the width of component c"""

    @abstractmethod
    def get_shape(self, c) -> callable:
        """Returns the shape function of component c"""

    def _get_display_content(self, circuit: ACircuit) -> str:
        return format_parameters(circuit.get_variables(), self.precision, self.nsimplify)
