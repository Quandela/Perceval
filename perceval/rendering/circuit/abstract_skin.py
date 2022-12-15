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

from abc import ABC, abstractmethod
from enum import Enum
from typing import Callable, Tuple
from multipledispatch import dispatch

from perceval.components import ACircuit, Processor, PERM
from perceval.components.abstract_component import AComponent
from perceval.components.non_unitary_components import TD


class ModeStyle(Enum):
    PHOTONIC = 0
    HERALD = 1
    DIGITAL = 2


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

    def __init__(self, stroke_style, style_subcircuit, compact_display: bool = False):
        self._compact = compact_display
        self.style = {ModeStyle.PHOTONIC: stroke_style,
                      ModeStyle.HERALD: {"stroke": None, "stroke_width": 1}
                      # ModeStyle.HERALD: {"stroke": "yellow", "stroke_width": 1}  # Use this for debug
                      }
        self.style_subcircuit = style_subcircuit

    @dispatch((ACircuit, TD), bool)
    def get_size(self, c: ACircuit, recursive: bool = False) -> Tuple[int, int]:
        """Gets the size of a circuit in arbitrary unit. If composite, it will take its components into account"""
        if not c.is_composite():
            return self.measure(c)

        # w represents the graph of the circuit.
        # Each value being the ouput of the rightmost component on the corresponding mode
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

    @dispatch(Processor, bool)
    def get_size(self, p: Processor, recursive: bool = False) -> Tuple[int, int]:
        height = p.m
        # w represents the graph of the circuit.
        # Each value being the ouput of the rightmost component on the corresponding mode
        w = [0] * p.circuit_size
        for modes, comp in p._components:
            if not isinstance(comp, PERM):
                height = max(height, comp.m + modes[0])

            r = slice(modes[0], modes[0] + comp.m)
            start_w = max(w[r])
            if comp.is_composite() and recursive:
                comp_width, _ = self.get_size(comp, False)
            else:
                comp_width = self.get_width(comp)
            end_w = start_w + comp_width
            w[r] = [end_w] * comp.m
        # For now, return the whole circuit height as processor height, even if some heralded modes are not shown.
        # TODO fix this
        return max(w), p.circuit_size  # min(p.circuit_size, height+2)

    def measure(self, c: AComponent) -> Tuple[int, int]:
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
    def get_shape(self, c) -> Callable:
        """Returns the shape function of component c"""
