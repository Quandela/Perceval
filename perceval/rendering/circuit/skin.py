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

from multipledispatch import dispatch

from perceval import ACircuit, Circuit
import perceval.lib.symb as symb
import perceval.lib.phys as phys



class ASkin(ABC):
    def __init__(self, compact_display: bool = False):
        self._compact = compact_display

    def get_size(self, c: ACircuit, recursive: bool = False):
        """Gets the size of a circuit. If composite, it will take its components into account"""
        if not c.is_composite():
            return self.measure(c)

        w = [0] * c.m
        for modes, comp in c._components:
            r = slice(modes[0], modes[0]+comp.m)
            start_w = max(w[r])
            if comp.is_composite() and recursive:
                comp_width, _ = self.get_size(comp)
            else:
                comp_width = self.get_width(comp)
            end_w = start_w + comp_width
            w[r] = [end_w] * comp.m

        return max(w), c.m

    def measure(self, c: ACircuit):
        """Returns the measure (in arbitrary unit (AU) where the space between two modes = 1 AU)
        of a single component treated as a block (meaning that a composite circuit will not be
        measured recursively. Use get_size() instead)
        """
        return self.get_width(c), c.m

    @abstractmethod
    def get_width(self, c) -> int:
        """Returns the width of component c"""
        pass


class SymbSkin(ASkin):
    def __init__(self, compact_display: bool = False):
        super().__init__(compact_display)

    @dispatch((Circuit, symb.Unitary, phys.Unitary))
    def get_width(self, c) -> int:
        return c.m

    @dispatch((symb.BS, symb.PBS))
    def get_width(self, c) -> int:
        w = 1 if self._compact else 2
        return w

    @dispatch((phys.BS, phys.PBS))
    def get_width(self, c) -> int:
        return 2

    @dispatch((symb.PS, phys.PS))
    def get_width(self, c) -> int:
        return 1

    @dispatch((symb.TD, phys.TD))
    def get_width(self, c) -> int:
        return 1

    @dispatch((symb.PERM, phys.PERM))
    def get_width(self, c) -> int:
        return 1

    @dispatch((symb.WP, phys.WP))
    def get_width(self, c) -> int:
        return 1

    @dispatch((symb.PR, phys.PR))
    def get_width(self, c) -> int:
        return 1
