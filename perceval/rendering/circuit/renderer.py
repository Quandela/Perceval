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
import copy
import math
from typing import Any, Tuple

from perceval.rendering.circuit import ASkin, ModeStyle
from perceval.rendering.format import Format
from perceval.rendering.canvas import Canvas, MplotCanvas, SvgCanvas, LatexCanvas
from perceval.components import ACircuit, Circuit, PortLocation, PERM, Herald, Barrier
from perceval.utils.format import format_parameters


class PortPos:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class ICircuitRenderer(ABC):
    """
    Base class for circuit renderers.
    Provides an interface to implement + a render_circuit() generic method.
    ICircuitRenderer internally works with circuit sizes in arbitrary units (AU), where single components and composite
    circuits size are measured by a given skin object.
    """

    def __init__(self, nsize):
        self._nsize = nsize  # number of modes
        self._mode_style = [ModeStyle.PHOTONIC] * nsize
        self._in_port_pos = []
        self._out_port_pos = []
        for i in range(nsize):
            self._in_port_pos.append(PortPos(0, i))
        for i in range(nsize):
            self._out_port_pos.append(PortPos(0, i))

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
                       map_param_kid: dict = None,
                       shift: int = 0,
                       recursive: bool = False,
                       precision: float = 1e-6,
                       nsimplify: bool = True) -> None:
        """Renders the input circuit
        """
        if not isinstance(circuit, Circuit):
            variables = circuit.get_variables(map_param_kid)
            description = format_parameters(variables, precision, nsimplify)
            self.append_circuit([p + shift for p in range(circuit.m)], circuit, description)

        if circuit.is_composite() and circuit.ncomponents() > 0:
            grouped_components = circuit.group_components_by_xgrid()
            for group in grouped_components:
                # each component of the group is to be rendered at the same horizontal position, use built-in
                # extend_pos method for that
                pos = None
                if len(group) > 1:
                    pos = -1
                    for r, _ in group:
                        pos = max(pos, self.max_pos(r[0], r[-1]))
                for r, c in group:
                    shiftr = [p + shift for p in r]
                    if c.is_composite() and c._components:
                        if recursive:
                            self._current_subblock_info = self._subblock_info.setdefault(c, {})
                            self.open_subblock(shiftr, c.name, self.get_circuit_size(c, recursive=True), c._color)
                            self.render_circuit(
                                c,
                                shift=shiftr[0],
                                map_param_kid=map_param_kid,
                                precision=precision,
                                nsimplify=nsimplify)
                            self.close_subblock(shiftr)
                        else:
                            component_vars = c.get_variables(map_param_kid)
                            description = format_parameters(component_vars, precision, nsimplify)
                            self.append_subcircuit(shiftr, c, description)
                    else:
                        component_vars = c.get_variables(map_param_kid)
                        description = format_parameters(component_vars, precision, nsimplify)
                        self.append_circuit(shiftr, c, description, pos=pos)
        self.extend_pos(0, circuit.m - 1)

    @abstractmethod
    def get_circuit_size(self, circuit: ACircuit, recursive: bool = False) -> Tuple[int, int]:
        """
        Returns the circuit size (in AU)
        """

    @abstractmethod
    def max_pos(self, start, end) -> int:
        """
        Returns the highest horizontal position on the circuit graph, between start and end modes (in AU)
        """

    @abstractmethod
    def extend_pos(self, start: int, end: int) -> None:
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
    def open_subblock(self, lines: Tuple[int, int], name: str, size: Tuple[int, int], color=None) -> None:
        """
        Opens a visual area, highlighting a part of the circuit
        """

    @abstractmethod
    def close_subblock(self, lines: Tuple[int, int]) -> None:
        """
        Close a visual area
        """

    @abstractmethod
    def draw(self) -> Any:
        """
        Finalize drawing, returns a fully drawn circuit (type is relative to the rendering method which was used).
        This should always be the last call.
        """

    @abstractmethod
    def append_subcircuit(self, lines: Tuple[int, int], circuit: Circuit, content: str) -> None:
        """
        Add a composite circuit to the rendering. Render each subcomponent independently.
        """

    @abstractmethod
    def append_circuit(self, lines: Tuple[int, int], circuit: ACircuit, content: str) -> None:
        """
        Add a component (or a circuit treated as a single component) to the rendering, on modes 'lines'
        """

    @abstractmethod
    def add_mode_index(self) -> None:
        """
        Render mode indexes on the right and left side of a previously rendered circuit
        """

    @abstractmethod
    def add_out_port(self, m: int, content: str, **opts) -> None:
        """
        Render a port on the right side (outputs) of a previously rendered circuit, located on mode 'm'
        """

    @abstractmethod
    def add_in_port(self, m: int, content: str, **opts) -> None:
        """
        Render a port on the left side (inputs) of a previously rendered circuit, located on mode 'm'
        """

    def set_herald_info(self, info):
        """
        Provides the renderer with a pre-computed dict mapping a component to
        information about which heralds are attached to its input and output
        ports. This is used to correctly position the Heralds within the
        circuit box, and not with the input and output ports.
        """
        pass

    @property
    def subblock_info(self) -> dict:
        """
        A dictionary mapping a subblock to a dictionary of settings relevant
        to its rendering.
        """
        return self._subblock_info


class TextRenderer(ICircuitRenderer):
    def __init__(self, nsize, hc=3, min_box_size=5):
        super().__init__(nsize)
        self._hc = hc
        self._h = [''] * (hc * nsize + 2)
        self.extend_pos(0, self._nsize - 1)
        self._depth = [0] * nsize
        self._offset = 0
        self.min_box_size = min_box_size

    def get_circuit_size(self, circuit: ACircuit, recursive: bool = False):
        return None  # Don't need circuit size for text rendering

    def open(self):
        for k in range(self._nsize):
            self._h[self._hc * k + 2] += "──"

    def close(self):
        self.extend_pos(0, self._nsize - 1)
        for k in range(self._nsize):
            self._h[self._hc * k + 2] += "──"

    def max_pos(self, start, end, header=False):
        maxpos = 0
        for nl in range(start * self._hc + (not header and 1 or 0), end * self._hc + 4 + (header and 1 or 0)):
            if len(self._h[nl]) > maxpos:
                maxpos = len(self._h[nl])
        return maxpos

    def extend_pos(self, start, end, internal=False, header=False, char=" ", pos=None):
        if pos is None:
            maxpos = self.max_pos(start, end, header)
        else:
            maxpos = pos
        for i in range(start * self._hc + (not header and 1 or 0), end * self._hc + 4 + ((header and not internal) and 1 or 0)):
            if internal:
                self._h[i] += char * (maxpos - len(self._h[i]))
            else:
                self._h[i] += ((i % self._hc) == 2 and "─" or char) * (maxpos - len(self._h[i]))

    def open_subblock(self, lines, name, size, color=None):
        start = lines[0]
        end = lines[-1]
        self.extend_pos(start, end, header=True)
        for k in range(start * self._hc, end * self._hc + 4):
            if k == start * self._hc:
                self._h[k] += "╔[" + name + "]"
            elif k % self._hc == 2:
                self._h[k] += "╫"
            else:
                self._h[k] += "║"
        self._h[end * self._hc + 4] += "╚"

    def close_subblock(self, lines):
        start = lines[0]
        end = lines[-1]
        self.extend_pos(start, end, header=True)
        for k in range(start * self._hc, end * self._hc + 4):
            if k == start * self._hc:
                self._h[k] += "╗"
            elif k % self._hc == 2:
                self._h[k] += "╫"
            else:
                self._h[k] += "║"
        self._h[end * self._hc + 4] += "╝"

    def append_subcircuit(self, lines, circuit, content):
        self.open_subblock(lines, circuit.name, None)
        self.extend_pos(lines[0], lines[-1], header=True, internal=True, char="░")
        self.close_subblock(lines)

    def append_circuit(self, lines, circuit, content, pos=None):
        # opening the box
        start = lines[0]
        end = lines[-1]
        self.extend_pos(start, end, pos=pos)
        
        if isinstance(circuit, Barrier):
            for k in range(start * self._hc + 1, (end + 1) * self._hc + 1):
                if k % self._hc == 2:
                    self._h[k] += "──║──"
                else:
                    self._h[k] += "  ║  "
            self.extend_pos(start, end, pos=pos)
            return

        # put variables on the right number of lines
        content = circuit.name + (content and "\n" + content or "")
        lcontents = content.split("\n")
        if start == end:
            content = " ".join(lcontents)
        else:
            nperlines = math.ceil((len(lcontents) - 1) / ((end - start) * self._hc))
            nlcontents = [lcontents[0]]
            idx = 1
            pnlcontent = []
            while idx < len(lcontents):
                pnlcontent.append(lcontents[idx])
                idx += 1
                if len(pnlcontent) == nperlines:
                    nlcontents.append(" ".join(pnlcontent))
                    pnlcontent = []
            if pnlcontent:
                nlcontents.append(" ".join(pnlcontent))
            content = "\n".join(nlcontents)

        # display box opening
        for k in range(start, end + 1):
            self._depth[k] += 1
        for k in range(start * self._hc + 1, end * self._hc + 3):
            if k == start * self._hc + 1:
                self._h[k] += "╭"
            elif k % self._hc == 2:
                self._h[k] += "┤"
            else:
                self._h[k] += "│"
        self._h[end * self._hc + 3] += "╰"

        lcontents = content.split("\n")
        maxw = max(len(nl) for nl in lcontents)
        maxw = max(maxw, self.min_box_size)
        # check if there are some "special effects" (centering _, right adjusting)
        for idx, l in enumerate(lcontents):
            if l.startswith("_"):
                lcontents[idx] = (" " * ((maxw - (len(l) - 1)) // 2)) + l[1:]

        for i in range(maxw):
            self._h[start * self._hc + 1] += "─"
            self._h[end * self._hc + 3] += "─"
            for j, l in enumerate(lcontents):
                if i < len(l):
                    self._h[self._hc * start + 2 + j] += l[i]
        self.extend_pos(start, end, True)
        # closing the box
        for k in range(start * self._hc + 1, end * self._hc + 3):
            if k == start * self._hc + 1:
                self._h[k] += "╮"
            elif k % self._hc == 2:
                self._h[k] += "├"
            else:
                self._h[k] += "│"
        self._h[end * self._hc + 3] += "╯"

    def draw(self):
        return "\n".join(self._h)

    def _set_offset(self, offset):
        offset_diff = offset - self._offset
        if offset_diff <= 0:
            return
        self._offset = offset
        for nl in range(len(self._h)):
            self._h[nl] = ' ' * offset_diff + self._h[nl]

    def add_mode_index(self):
        offset = len(str(self._nsize)) + 1
        self._set_offset(offset)
        for k in range(self._nsize):
            self._h[self._hc * k + 2] = f'{k:{offset-1}d}:' + self._h[self._hc * k + 2][offset:]
            self._h[self._hc * k + 2] += ':' + str(k) + f" (depth {self._depth[k]})"

    def add_out_port(self, n_mode, port, **opts):
        content = ''
        if isinstance(port, Herald):
            content = port.expected
        for i in range(port.m):
            self._h[self._hc * (n_mode + i) + 2] += f'[{content})'
            self._h[self._hc * (n_mode + i) + 3] += f"[{port.name}]"

    def add_in_port(self, n_mode, port, **opts):
        content = ''
        if isinstance(port, Herald):
            content = str(port.expected)
        shape_size = len(content) + 2
        name = port.name
        name_size = len(name)
        self._set_offset(max(shape_size, name_size))
        for i in range(port.m):
            self._h[self._hc * (n_mode + i) + 2] = f'({content}]' + \
                '─' * (self._offset - shape_size) + \
                self._h[self._hc * (n_mode + i) + 2][self._offset:]
            self._h[self._hc * (n_mode + i) + 3] = name + ' ' * \
                (self._offset - name_size) + \
                self._h[self._hc * (n_mode + i) + 3][self._offset:]


class CanvasRenderer(ICircuitRenderer):

    AFFIX_PORT_SIZE = 15
    AFFIX_ALL_SIZE = 25
    SCALE = 50

    def __init__(self, nsize, canvas: Canvas, skin):
        super().__init__(nsize)
        # first position available for row n
        self._chart = [0] * (nsize + 1)

        self._canvas = canvas
        self._skin = skin
        self._canvas.set_offset(
            (0, 0),
            CanvasRenderer.AFFIX_ALL_SIZE,
            CanvasRenderer.SCALE * (nsize + 1))
        self._n_font_size = min(10, max(6, self._nsize + 1))

        self._herald_info = None

    def open(self):
        for k in range(self._nsize):
            mode_style = self._skin.style[self._mode_style[k]]
            if mode_style['stroke']:
                self._canvas.add_mpath([
                    "M",
                    CanvasRenderer.AFFIX_ALL_SIZE - CanvasRenderer.AFFIX_PORT_SIZE,
                    CanvasRenderer.SCALE / 2 + CanvasRenderer.SCALE * k,
                    "l",
                    CanvasRenderer.AFFIX_PORT_SIZE, 0], **mode_style)

    def get_circuit_size(self, circuit: ACircuit, recursive: bool = False):
        return self._skin.get_size(circuit, recursive)

    def add_mode_index(self):
        for k in range(self._nsize):
            if self._mode_style[k] != ModeStyle.HERALD:
                self._canvas.add_text(
                    (CanvasRenderer.AFFIX_ALL_SIZE,
                     CanvasRenderer.SCALE / 2 + 3 + CanvasRenderer.SCALE * k),
                    str(k),
                    self._n_font_size,
                    ta="right")

        self._canvas.set_offset(
            (0, 0),
            CanvasRenderer.AFFIX_ALL_SIZE,
            CanvasRenderer.SCALE * (self._nsize + 1))
        for k in range(self._nsize):
            if self._mode_style[k] != ModeStyle.HERALD:
                self._canvas.add_text(
                    (
                        0,
                        CanvasRenderer.SCALE / 2 + 3 + CanvasRenderer.SCALE * k
                    ),
                    str(k),
                    self._n_font_size,
                    ta="left")

    def add_out_port(self, n_mode, port, **opts):
        max_pos = max(self._chart[0:self._nsize])
        h_pos = self._out_port_pos[n_mode].x
        v_pos = self._out_port_pos[n_mode].y
        self._canvas.set_offset(
            (
                CanvasRenderer.AFFIX_ALL_SIZE + CanvasRenderer.SCALE * (h_pos or max_pos),
                CanvasRenderer.SCALE * v_pos
            ),
            CanvasRenderer.AFFIX_ALL_SIZE,
            CanvasRenderer.SCALE)
        opts['starting_mode'] = n_mode
        self._canvas.add_shape(
            self._skin.get_shape(
                port, PortLocation.OUTPUT), port, None, None, **opts)

    def add_in_port(self, n_mode, port, **opts):
        h_pos = self._in_port_pos[n_mode].x * CanvasRenderer.SCALE
        v_pos = self._in_port_pos[n_mode].y * CanvasRenderer.SCALE
        self._canvas.set_offset(
            (h_pos, v_pos),
            CanvasRenderer.AFFIX_ALL_SIZE,
            CanvasRenderer.SCALE)
        opts['starting_mode'] = n_mode
        self._canvas.add_shape(
            self._skin.get_shape(port, PortLocation.INPUT),
            port,
            None,
            None,
            **opts)

    def open_subblock(self, lines, name, size, color=None):
        # Get recommended margins for this block
        margins = self._current_subblock_info.get('margins', (0, 0))

        start = lines[0]
        end = lines[-1]
        subblock_start = self.max_pos(start, end)
        area = (
            subblock_start,
            start,
            size[0] + margins[0] + margins[1],
            size[1])
        self._canvas.set_offset(
            (
                CanvasRenderer.AFFIX_ALL_SIZE + CanvasRenderer.SCALE * area[0],
                CanvasRenderer.SCALE * area[1]
            ),
            CanvasRenderer.SCALE * area[2],
            CanvasRenderer.SCALE * area[3])
        if color is None:
            color = "lightblue"
        self._canvas.set_background_color(color)
        self._canvas.add_rect(
            (2, 2),
            CanvasRenderer.SCALE * area[2] - 4,
            CanvasRenderer.SCALE * area[3] - 4,
            fill=color,
            stroke_dasharray="1,2")
        self._canvas.add_text(
            (4, CanvasRenderer.SCALE * (end - start + 1) + 5),
            name.upper(), 8)
        # Extend lines on the left side
        if margins[0]:
            self.extend_pos(start, end, self.max_pos(start, end) + margins[0])

    def close_subblock(self, lines):
        start = lines[0]
        end = lines[-1]
        subblock_end = self.max_pos(start, end)
        # Extend lines on the right side
        right_margins = self._current_subblock_info.get('margins', (0, 0))[1]
        self.extend_pos(start, end, subblock_end + right_margins)

    def max_pos(self, start, end, _=None):
        return max(self._chart[start:end + 1])

    def extend_pos(self, start, end, pos=None):
        if pos is None:
            maxpos = self.max_pos(start, end)
        else:
            maxpos = pos
        for p in range(start, end + 1):
            if self._chart[p] != maxpos:
                self._canvas.set_offset(
                    (
                        CanvasRenderer.AFFIX_ALL_SIZE + self._chart[p] * CanvasRenderer.SCALE,
                        p * CanvasRenderer.SCALE
                    ),
                    (maxpos - self._chart[p]) * CanvasRenderer.SCALE,
                    CanvasRenderer.SCALE)
                style = self._skin.style[self._mode_style[p]]
                if style['stroke']:
                    self._canvas.add_mline(
                        [
                            0,
                            CanvasRenderer.SCALE / 2,
                            (maxpos - self._chart[p]) * CanvasRenderer.SCALE,
                            CanvasRenderer.SCALE / 2
                        ],
                        **style)
            self._chart[p] = maxpos

    def _add_shape(self, lines, circuit, content, w, shape_fn=None, pos=None):
        if shape_fn is None:
            shape_fn = self._skin.get_shape(circuit)
        start = lines[0]
        end = lines[-1]
        self.extend_pos(start, end, pos=pos)
        max_pos = self.max_pos(start, end)
        self._canvas.set_offset(
            (
                CanvasRenderer.AFFIX_ALL_SIZE + CanvasRenderer.SCALE * max_pos,
                CanvasRenderer.SCALE * start
            ),
            CanvasRenderer.SCALE * w,
            CanvasRenderer.SCALE * (end - start + 1))
        modes = self._mode_style[start:(end + 1)]
        self._canvas.add_shape(shape_fn, circuit, content, modes)

    def set_herald_info(self, info):
        self._herald_info = info

    def _update_mode_style(self, lines, circuit, w: int):
        if not isinstance(circuit, PERM):
            input_heralds = {}
            output_heralds = {}

            herald_info = self._herald_info if self._herald_info else {}
            if circuit in herald_info:
                output_heralds = herald_info[circuit].output_heralds
                input_heralds = herald_info[circuit].input_heralds

            # Position input and output heralds
            for in_mode, herald_in_mode in input_heralds.items():
                self._in_port_pos[herald_in_mode].y = lines[0] + in_mode
                self._in_port_pos[herald_in_mode].x = \
                    self._chart[lines[0] + in_mode]
                # Start drawing this mode in "photonic" style
                self._mode_style[lines[0] + in_mode] = ModeStyle.PHOTONIC
            for out_mode, herald_out_mode in output_heralds.items():
                self._out_port_pos[herald_out_mode].y = lines[0] + out_mode
                self._out_port_pos[herald_out_mode].x = \
                    self._chart[lines[0] + out_mode] + w
                # Stop drawing this mode (set it in "herald" style)
                self._mode_style[lines[0] + out_mode] = ModeStyle.HERALD
        if isinstance(circuit, PERM):
            m0 = lines[0]
            out_modes = copy.copy(self._mode_style)
            for m_input, m_output in enumerate(circuit.perm_vector):
                out_modes[m_output + lines[0]] = self._mode_style[m_input + m0]
            self._mode_style = out_modes

    def append_circuit(self, lines, circuit, content, pos=None):
        w = self._skin.get_width(circuit)
        if w:
            self._add_shape(lines, circuit, content, w, pos=pos)
            self._update_mode_style(lines, circuit, w)
            for i in range(lines[0], lines[-1] + 1):
                self._chart[i] += w

    def append_subcircuit(self, lines, circuit, content):
        w = self._skin.style_subcircuit['width']
        if w:
            self._add_shape(
                lines, circuit, content, w, self._skin.subcircuit_shape)
            self._update_mode_style(lines, circuit, w)
            for i in range(lines[0], lines[-1] + 1):
                self._chart[i] += w

    def close(self):
        self.extend_pos(0, self._nsize - 1)
        max_pos = self.max_pos(0, self._nsize - 1)
        self._canvas.set_offset(
            (CanvasRenderer.AFFIX_ALL_SIZE + CanvasRenderer.SCALE * max_pos, 0),
            CanvasRenderer.AFFIX_ALL_SIZE,
            CanvasRenderer.SCALE * (self._nsize + 1))
        for k in range(self._nsize):
            mode_style = self._skin.style[self._mode_style[k]]
            if mode_style['stroke']:
                self._canvas.add_mpath([
                    "M",
                    0,
                    CanvasRenderer.SCALE / 2 + CanvasRenderer.SCALE * k,
                    "l",
                    CanvasRenderer.AFFIX_PORT_SIZE, 0],
                    **mode_style)

    def draw(self):
        return self._canvas.draw()


class PreRenderer(ICircuitRenderer):
    """
    This performs a dummy rendering pass to keep track of potential
    layout issues to be fixed in the main rendering.

    At the moment it is keeping track of the recommended margins
    for each subblock.
    """
    def __init__(self, nsize, skin):
        super().__init__(nsize)
        self._chart = [0] * (nsize + 1)
        self._herald_info = None
        self._skin = skin

        # All these are relative to the subblock currently being rendered
        self._herald_range = [0, 0]
        self._subblock_start = 0

    def open(self):
        pass

    def draw(self):
        pass

    def close(self):
        pass

    def add_mode_index(self) -> None:
        pass

    def add_out_port(self, m: int, content: str, **opts) -> None:
        pass

    def add_in_port(self, m: int, content: str, **opts) -> None:
        pass

    def set_herald_info(self, info):
        self._herald_info = info

    def get_circuit_size(self, circuit: ACircuit, recursive: bool = False):
        return None

    def open_subblock(self, lines, name, size, color=None):
        start = lines[0]
        end = lines[-1]
        self._subblock_start = self.max_pos(start, end)
        self._herald_range = [1 << 32, -1]

    def close_subblock(self, lines):
        start = lines[0]
        end = lines[-1]
        subblock_end = self.max_pos(start, end)
        # Add the margin requirements for this subblock
        self._current_subblock_info['margins'] = (
            int(self._herald_range[0] == self._subblock_start),
            int(self._herald_range[1] == subblock_end))

    def max_pos(self, start, end, _=None):
        return max(self._chart[start:end + 1])

    def extend_pos(self, start, end, pos=None):
        if pos is None:
            maxpos = self.max_pos(start, end)
        else:
            maxpos = pos
        for p in range(start, end + 1):
            self._chart[p] = maxpos

    def _add_shape(self, lines, circuit, content, w, shape_fn=None, pos=None):
        start = lines[0]
        end = lines[-1]
        self.extend_pos(start, end, pos=pos)

    def _update_mode_style(self, lines, circuit, w: int):
        if not isinstance(circuit, PERM):
            input_heralds = {}
            output_heralds = {}
            herald_info = self._herald_info if self._herald_info else {}
            if circuit in herald_info:
                output_heralds = herald_info[circuit].output_heralds
                input_heralds = herald_info[circuit].input_heralds

            for out_mode, herald_out_mode in output_heralds.items():
                self._herald_range[1] = max(
                    self._herald_range[1],
                    self._chart[lines[0] + out_mode] + w)
            for in_mode, herald_in_mode in input_heralds.items():
                self._herald_range[0] = min(
                    self._herald_range[0],
                    self._chart[lines[0] + in_mode])

    def append_circuit(self, lines, circuit, content, pos=None):
        w = self._skin.get_width(circuit)
        if w:
            self._add_shape(lines, circuit, content, w, pos=pos)
            self._update_mode_style(lines, circuit, w)
            for i in range(lines[0], lines[-1] + 1):
                self._chart[i] += w

    def append_subcircuit(self, lines, circuit, content):
        w = self._skin.style_subcircuit['width']
        if w:
            self._add_shape(
                lines, circuit, content, w, self._skin.subcircuit_shape)
            self._update_mode_style(lines, circuit, w)
            for i in range(lines[0], lines[-1] + 1):
                self._chart[i] += w


def create_renderer(
    n: int,  # number of modes
    output_format: Format = Format.TEXT,  # rendering method
    skin: ASkin = None,  # skin (unused in text rendering)
    **opts
) -> ICircuitRenderer:
    """
    Creates a renderer given the selected format. Dispatches parameters to generated canvas objects
    A skin object is needed for circuit graphic rendering.

    This returns a (renderer, pre_renderer) tuple. It is recommended to
    invoke the pre-renderer on the circuit to correctly pre-compute
    additional position information that cannot be guessed in a single pass.
    """
    if output_format == Format.TEXT:
        return TextRenderer(n), None

    assert skin is not None, "A skin must be selected for circuit graphical rendering"
    if output_format == Format.HTML:
        canvas = SvgCanvas(**opts)
    elif output_format == Format.LATEX:
        canvas = LatexCanvas(**opts)
    else:
        canvas = MplotCanvas(**opts)
    return CanvasRenderer(n, canvas, skin), PreRenderer(n, skin)
