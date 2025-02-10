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

from perceval.utils import ModeType
from perceval.components import ACircuit, Circuit, APort


class ICircuitRenderer(ABC):
    """
    Base class for circuit renderers.
    Provides an interface to implement + a render_circuit() generic method.
    ICircuitRenderer internally works with circuit sizes in arbitrary units (AU), where single components and composite
    circuits size are measured by a given skin object.
    """

    def __init__(self, nsize):
        self._nsize = nsize  # number of modes
        self._mode_style = [ModeType.PHOTONIC] * nsize

        # A dictionary mapping a subblock to information pertaining to its
        # rendering. This is written by the pre-rendering pass, and read by
        # the main rendering pass.
        self._subblock_info = {}

        # Custom settings for the subblock being rendered.
        # They are loaded before open_subblock is invoked.
        self._current_subblock_info = {}

    def set_mode_style(self, index, style):
        self._mode_style[index] = style

    def render_circuit(self,
                       circuit: ACircuit,
                       shift: int = 0,
                       recursive: bool = False,
                       precision: float = 1e-6,
                       nsimplify: bool = True) -> None:
        """Renders the input circuit"""
        if not isinstance(circuit, Circuit):
            self.append_circuit(tuple(p + shift for p in range(circuit.m)), circuit)

        if circuit.is_composite() and circuit.ncomponents() > 0:
            for r, c in circuit._components:
                shiftr = tuple(p + shift for p in r)
                if c.is_composite():
                    if c._components:
                        if recursive:
                            self._current_subblock_info = self._subblock_info.setdefault(c, {})
                            self.open_subblock(shiftr, c.name, self.get_circuit_size(c, recursive=False), c._color)
                            self.render_circuit(
                                c,
                                shift=shiftr[0],
                                precision=precision,
                                nsimplify=nsimplify)
                            self.close_subblock(shiftr)
                        else:
                            self.append_subcircuit(shiftr, c)
                else:
                    self.append_circuit(shiftr, c)
        self.extend_pos(0, circuit.m - 1)

    @abstractmethod
    def get_circuit_size(self, circuit: ACircuit, recursive: bool = False) -> tuple[int, int]:
        """
        Returns the circuit size (in AU)
        """

    @abstractmethod
    def max_pos(self, start, end) -> int:
        """
        Returns the highest horizontal position on the circuit graph, between start and end modes (in AU)
        """

    @abstractmethod
    def extend_pos(self, start: int, end: int, margin: int = 0) -> None:
        """
        Extends horizontal position on the circuit graph, from modes 'start' to 'end'
        """

    @abstractmethod
    def open(self) -> None:
        """
        Starts the circuit drawing
        """

    @abstractmethod
    def close(self) -> None:
        """
        Finalizes circuit rendering when nothing more needs to be added.
        The opposite 'open' action should be run in __init__.
        """

    @abstractmethod
    def open_subblock(self, lines: tuple[int, ...], name: str, size: tuple[int, int], color=None) -> None:
        """
        Opens a visual area, highlighting a part of the circuit
        """

    @abstractmethod
    def close_subblock(self, lines: tuple[int, ...]) -> None:
        """
        Close a visual area
        """

    @abstractmethod
    def draw(self) -> any:
        """
        Finalize drawing, returns a fully drawn circuit (type is relative to the rendering method which was used).
        This should always be the last call.
        """

    @abstractmethod
    def append_subcircuit(self, lines: tuple[int, ...], circuit: Circuit) -> None:
        """
        Add a composite circuit to the rendering. Render each subcomponent independently.
        """

    @abstractmethod
    def append_circuit(self, lines: tuple[int, ...], circuit: ACircuit) -> None:
        """
        Add a component (or a circuit treated as a single component) to the rendering, on modes 'lines'
        """

    @abstractmethod
    def add_mode_index(self) -> None:
        """
        Render mode indexes on the right and left side of a previously rendered circuit
        """

    @abstractmethod
    def display_input_photons(self, input_pos) -> None:
        """
        Display photons on input modes
        """

    @abstractmethod
    def add_out_port(self, m: int, port: APort) -> None:
        """
        Render a port on the right side (outputs) of a previously rendered circuit, located on mode 'm'
        """

    @abstractmethod
    def add_in_port(self, m: int, port: APort) -> None:
        """
        Render a port on the left side (inputs) of a previously rendered circuit, located on mode 'm'
        """

    @abstractmethod
    def add_detectors(self, detector_list: list) -> None:
        """
        Render detectors when they exist
        """

    def set_herald_info(self, info):
        """
        Provides the renderer with a pre-computed dict mapping a component to
        information about which heralds are attached to its input and output
        ports. This is used to correctly position the Heralds within the
        circuit box, and not with the input and output ports.
        """

    @property
    def subblock_info(self) -> dict:
        """
        A dictionary mapping a subblock to a dictionary of settings relevant
        to its rendering.
        """
        return self._subblock_info
