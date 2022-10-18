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
import copy
import math
from typing import Any, Tuple

from perceval.rendering.circuit import ASkin, ModeStyle
from perceval.rendering.format import Format
from perceval.rendering.canvas import Canvas, MplotCanvas, SvgCanvas
from perceval.components import ACircuit, Circuit, PortLocation, PERM, Herald
from perceval.utils.format import format_parameters


class PortPos:
    def __init__(self, x, y, fixed=True):
        self.x = x
        self.y = y
        self.fixed = fixed
        self._initial_mode = y

    @property
    def initial_mode(self):
        return self._initial_mode


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

    def set_mode_style(self, index, style):
        self._mode_style[index] = style
        if style == ModeStyle.HERALD:
            self._in_port_pos[index].fixed = False

    def render_circuit(self,
                       circuit: ACircuit,
                       map_param_kid: dict = None,
                       shift: int = 0,
                       recursive: bool = False,
                       precision: float = 1e-6,
                       nsimplify: bool = True):
        """
        Renders the input circuit
        """
        if not isinstance(circuit, Circuit):
            variables = circuit.get_variables(map_param_kid)
            description = format_parameters(variables, precision, nsimplify)
            self.append_circuit([p + shift for p in range(circuit.m)], circuit, description)

        if circuit.is_composite() and circuit.ncomponents() > 0:
            for _, (r, c) in enumerate(circuit._components):
                shiftr = [p+shift for p in r]
                if c.is_composite() and c._components:
                    if recursive:
                        self.open_subblock(shiftr, c.name, self.get_circuit_size(c, recursive=True), c._color)
                        self.render_circuit(c, shift=shiftr[0], map_param_kid=map_param_kid,
                                            precision=precision, nsimplify=nsimplify)
                        self.close_subblock(shiftr)
                    else:
                        component_vars = c.get_variables(map_param_kid)
                        description = format_parameters(component_vars, precision, nsimplify)
                        self.append_subcircuit(shiftr, c, description)
                else:
                    component_vars = c.get_variables(map_param_kid)
                    description = format_parameters(component_vars, precision, nsimplify)
                    self.append_circuit(shiftr, c, description)

        self.extend_pos(0, circuit.m - 1)

    @abstractmethod
    def get_circuit_size(self, circuit: ACircuit, recursive: bool = False) -> Tuple[int, int]:
        """
        Returns the circuit size (in AU)
        """

    @abstractmethod
    def max_pos(self, start, end, header) -> int:
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

    def set_out_herald_info(self, info):
        """
        Handles a very specific need for canvas rendering: moving out heralds as left as possible in the displayed
        processor. See usage in CanvasRenderer
        """
        pass


class TextRenderer(ICircuitRenderer):
    def __init__(self, nsize, hc=3, min_box_size=5):
        super().__init__(nsize)
        self._hc = hc
        self._h = ['']*(hc*nsize+2)
        self.extend_pos(0, self._nsize-1)
        self._depth = [0]*nsize
        self._offset = 0
        self.min_box_size = min_box_size

    def get_circuit_size(self, circuit: ACircuit, recursive: bool = False):
        return None  # Don't need circuit size for text rendering

    def open(self):
        for k in range(self._nsize):
            self._h[self._hc*k+2] += "──"

    def close(self):
        self.extend_pos(0, self._nsize-1)
        for k in range(self._nsize):
            self._h[self._hc*k+2] += "──"

    def max_pos(self, start, end, header):
        maxpos = 0
        for nl in range(start*self._hc+(not header and 1 or 0), end*self._hc+4+(header and 1 or 0)):
            if len(self._h[nl]) > maxpos:
                maxpos = len(self._h[nl])
        return maxpos

    def extend_pos(self, start, end, internal=False, header=False, char=" "):
        maxpos = self.max_pos(start, end, header)
        for i in range(start*self._hc+(not header and 1 or 0), end*self._hc+4+((header and not internal) and 1 or 0)):
            if internal:
                self._h[i] += char*(maxpos-len(self._h[i]))
            else:
                self._h[i] += ((i % self._hc) == 2 and "─" or char)*(maxpos-len(self._h[i]))

    def open_subblock(self, lines, name, size, color=None):
        start = lines[0]
        end = lines[-1]
        self.extend_pos(start, end, header=True)
        for k in range(start*self._hc, end*self._hc+4):
            if k == start*self._hc:
                self._h[k] += "╔["+name+"]"
            elif k % self._hc == 2:
                self._h[k] += "╫"
            else:
                self._h[k] += "║"
        self._h[end*self._hc+4] += "╚"

    def close_subblock(self, lines):
        start = lines[0]
        end = lines[-1]
        self.extend_pos(start, end, header=True)
        for k in range(start*self._hc, end*self._hc+4):
            if k == start*self._hc:
                self._h[k] += "╗"
            elif k % self._hc == 2:
                self._h[k] += "╫"
            else:
                self._h[k] += "║"
        self._h[end*self._hc+4] += "╝"

    def append_subcircuit(self, lines, circuit, content):
        self.open_subblock(lines, circuit.name, None)
        self.extend_pos(lines[0], lines[-1], header=True, internal=True, char="░")
        self.close_subblock(lines)

    def append_circuit(self, lines, circuit, content):
        # opening the box
        start = lines[0]
        end = lines[-1]
        self.extend_pos(start, end)
        # put variables on the right number of lines
        content = circuit.name + (content and "\n"+content or "")
        lcontents = content.split("\n")
        if start == end:
            content = " ".join(lcontents)
        else:
            nperlines = math.ceil((len(lcontents)-1)/((end-start)*self._hc))
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
        for k in range(start, end+1):
            self._depth[k] += 1
        for k in range(start*self._hc+1, end*self._hc+3):
            if k == start*self._hc+1:
                self._h[k] += "╭"
            elif k % self._hc == 2:
                self._h[k] += "┤"
            else:
                self._h[k] += "│"
        self._h[end*self._hc+3] += "╰"

        lcontents = content.split("\n")
        maxw = max(len(nl) for nl in lcontents)
        maxw = max(maxw, self.min_box_size)
        # check if there are some "special effects" (centering _, right adjusting)
        for idx, l in enumerate(lcontents):
            if l.startswith("_"):
                lcontents[idx] = (" " * ((maxw-(len(l)-1))//2)) + l[1:]

        for i in range(maxw):
            self._h[start*self._hc+1] += "─"
            self._h[end*self._hc+3] += "─"
            for j, l in enumerate(lcontents):
                if i < len(l):
                    self._h[self._hc*start+2+j] += l[i]
        self.extend_pos(start, end, True)
        # closing the box
        for k in range(start*self._hc+1, end*self._hc+3):
            if k == start*self._hc+1:
                self._h[k] += "╮"
            elif k % self._hc == 2:
                self._h[k] += "├"
            else:
                self._h[k] += "│"
        self._h[end*self._hc+3] += "╯"

    def draw(self):
        return "\n".join(self._h)

    def _set_offset(self, offset):
        offset_diff = offset - self._offset
        if offset_diff <= 0:
            return
        self._offset = offset
        for nl in range(len(self._h)):
            self._h[nl] = ' '*offset_diff + self._h[nl]

    def add_mode_index(self):
        offset = len(str(self._nsize))+1
        self._set_offset(offset)
        for k in range(self._nsize):
            self._h[self._hc*k + 2] = f'{k:{offset-1}d}:' + self._h[self._hc*k + 2][offset:]
            self._h[self._hc*k + 2] += ':' + str(k) + f" (depth {self._depth[k]})"

    def add_out_port(self, n_mode, port, **opts):
        content = ''
        if isinstance(port, Herald):
            content = port.expected
        for i in range(port.m):
            self._h[self._hc*(n_mode+i) + 2] += f'[{content})'
            self._h[self._hc*(n_mode+i) + 3] += f"[{port.name}]"

    def add_in_port(self, n_mode, port, **opts):
        content = ''
        if isinstance(port, Herald):
            content = str(port.expected)
        shape_size = len(content) + 2
        name = port.name
        name_size = len(name)
        self._set_offset(max(shape_size, name_size))
        for i in range(port.m):
            self._h[self._hc*(n_mode+i) + 2] = f'({content}]' + '─'*(self._offset-shape_size) \
                                      + self._h[self._hc*(n_mode+i) + 2][self._offset:]
            self._h[self._hc*(n_mode+i) + 3] = name + ' '*(self._offset-name_size) + \
                                               self._h[self._hc*(n_mode+i) + 3][self._offset:]


class CanvasRenderer(ICircuitRenderer):
    affix_port_size = 15
    affix_all_size = 25

    def __init__(self, nsize, canvas: Canvas, skin):
        super().__init__(nsize)
        # first position available for row n
        self._chart = [0] * (nsize+1)
        self._canvas = canvas
        self._skin = skin
        self._canvas.set_offset((0, 0),
                                CanvasRenderer.affix_all_size, 50 * (nsize + 1))
        self._n_font_size = min(10, max(6, self._nsize+1))

        self._out_herald_info = None

    def open(self):
        for k in range(self._nsize):
            mode_style = self._skin.style[self._mode_style[k]]
            if mode_style['stroke']:
                self._canvas.add_mpath(["M", CanvasRenderer.affix_all_size-CanvasRenderer.affix_port_size, 25 + 50 * k,
                                        "l", CanvasRenderer.affix_port_size, 0], **mode_style)

    def get_circuit_size(self, circuit: ACircuit, recursive: bool = False):
        return self._skin.get_size(circuit, recursive)

    def add_mode_index(self):
        for k in range(self._nsize):
            if self._mode_style[k] != ModeStyle.HERALD:
                self._canvas.add_text((CanvasRenderer.affix_all_size, 28 + 50 * k), str(k), self._n_font_size, ta="right")

        self._canvas.set_offset((0, 0), CanvasRenderer.affix_all_size, 50 * (self._nsize + 1))
        for k in range(self._nsize):
            if self._mode_style[k] != ModeStyle.HERALD:
                self._canvas.add_text((0, 28 + 50 * k), str(k), self._n_font_size, ta="left")

    def add_out_port(self, n_mode, port, **opts):
        max_pos = max(self._chart[0:self._nsize])
        h_pos = self._out_port_pos[n_mode].x
        v_pos = self._out_port_pos[n_mode].y
        self._canvas.set_offset((CanvasRenderer.affix_all_size + 50*(h_pos or max_pos), 50*v_pos),
                                CanvasRenderer.affix_all_size, 50)
        opts['starting_mode'] = n_mode
        self._canvas.add_shape(self._skin.get_shape(port, PortLocation.OUTPUT), port, None, None, **opts)

    def add_in_port(self, n_mode, port, **opts):
        h_pos = self._in_port_pos[n_mode].x*50
        v_pos = self._in_port_pos[n_mode].y*50
        self._canvas.set_offset((h_pos, v_pos), CanvasRenderer.affix_all_size, 50)
        opts['starting_mode'] = n_mode
        self._canvas.add_shape(self._skin.get_shape(port, PortLocation.INPUT), port, None, None, **opts)

    def open_subblock(self, lines, name, size, color=None):
        start = lines[0]
        end = lines[-1]

        self.extend_pos(start, end)
        area = (self.max_pos(start, end), start, size[0], size[1])
        self._canvas.set_offset((CanvasRenderer.affix_all_size + 50 * area[0], 50 * area[1]),
                                50 * area[2], 50 * area[3])
        if color is None:
            color = "lightblue"
        self._canvas.add_rect((2, 2), 50 * area[2] - 4, 50 * area[3] - 4, fill=color, stroke_dasharray="1,2")
        self._canvas.add_text((4, 50 * (end - start + 1) + 5), name.upper(), 8)

    def close_subblock(self, lines):
        start = lines[0]
        end = lines[-1]
        self.extend_pos(start, end)

    def max_pos(self, start, end, _=None):
        return max(self._chart[start:end+1])

    def extend_pos(self, start, end):
        maxpos = self.max_pos(start, end)
        for p in range(start, end+1):
            if self._chart[p] != maxpos:
                self._canvas.set_offset((CanvasRenderer.affix_all_size+self._chart[p]*50, p*50),
                                        (maxpos-self._chart[p])*50, 50)
                style = self._skin.style[self._mode_style[p]]
                if style['stroke']:
                    self._canvas.add_mline([0, 25, (maxpos-self._chart[p])*50, 25], **style)
            self._chart[p] = maxpos

    def _add_shape(self, lines, circuit, content, w, shape_fn=None):
        if shape_fn is None:
            shape_fn = self._skin.get_shape(circuit)
        start = lines[0]
        end = lines[-1]
        self.extend_pos(start, end)
        max_pos = self.max_pos(start, end)
        self._canvas.set_offset((CanvasRenderer.affix_all_size + 50 * max_pos, 50 * start), 50 * w,
                                50 * (end - start + 1))

        modes = self._mode_style[start:(end + 1)]
        self._canvas.add_shape(shape_fn, circuit, content, modes)

    def set_out_herald_info(self, hinf):
        self._out_herald_info = hinf

    def _search_component_in_out_herald_info(self, c):
        if self._out_herald_info is None:
            return None, None
        for k, v in self._out_herald_info.items():
            if c is v[1]:
                return k, v[0]
        return None, None

    def _update_mode_style(self, lines, circuit, w: int, subc_mode: bool = False):
        # BEGIN Mode tracking + herald positionning algo
        m0 = lines[0]
        for i in lines:

            ppos = None
            for p in self._in_port_pos:
                if p.y == i and p.fixed is False:
                    ppos = p
                    break

            if ppos is not None:
                if not isinstance(circuit, PERM):
                    ppos.x = self._chart[i]
                    ppos.fixed = True
                    if subc_mode:
                        self._out_port_pos[ppos.initial_mode].x = ppos.x + w
                        self._out_port_pos[ppos.initial_mode].y = ppos.y
                    else:
                        self._mode_style[i] = ModeStyle.PHOTONIC

                else:
                    ppos.y = circuit.perm_vector[ppos.y - lines[0]] + lines[0]
                    ppos.fixed = None

        # Out heralds
        if not isinstance(circuit, PERM) and not subc_mode:
            herald_out_mode, component_out_mode = self._search_component_in_out_herald_info(circuit)
            if herald_out_mode is not None:
                self._mode_style[lines[0] + component_out_mode] = ModeStyle.HERALD
                self._out_port_pos[herald_out_mode].y = lines[0] + component_out_mode
                self._out_port_pos[herald_out_mode].x = self._chart[lines[0] + component_out_mode] + w

        if isinstance(circuit, PERM):
            out_modes = copy.copy(self._mode_style)
            for m_input, m_output in enumerate(circuit.perm_vector):
                out_modes[m_output+m0] = self._mode_style[m_input+m0]
            self._mode_style = out_modes

        for p in self._in_port_pos:
            if p.fixed is None:
                p.fixed = False
        # END Mode tracking + herald positionning algo

    def append_circuit(self, lines, circuit, content):
        w = self._skin.get_width(circuit)
        self._add_shape(lines, circuit, content, w)
        self._update_mode_style(lines, circuit, w)
        for i in range(lines[0], lines[-1] + 1):
            self._chart[i] += w

    def append_subcircuit(self, lines, circuit, content):
        w = self._skin.style_subcircuit['width']
        self._add_shape(lines, circuit, content, w, self._skin.subcircuit_shape)
        self._update_mode_style(lines, circuit, w, True)
        for i in range(lines[0], lines[-1] + 1):
            self._chart[i] += w

    def close(self):
        self.extend_pos(0, self._nsize - 1)
        max_pos = self.max_pos(0, self._nsize-1)
        self._canvas.set_offset((CanvasRenderer.affix_all_size+50*max_pos, 0),
                                CanvasRenderer.affix_all_size, 50*(self._nsize+1))
        for k in range(self._nsize):
            mode_style = self._skin.style[self._mode_style[k]]
            if mode_style['stroke']:
                self._canvas.add_mpath(["M", 0, 25 + 50 * k,
                                        "l", CanvasRenderer.affix_port_size, 0], **mode_style)

    def draw(self):
        return self._canvas.draw()


def create_renderer(
        n: int,  # number of modes
        output_format: Format = Format.TEXT,  # rendering method
        skin: ASkin = None,  # skin (unused in text rendering)
        **opts
) -> ICircuitRenderer:
    """
    Creates a renderer given the selected format. Dispatches parameters to generated canvas objects
    A skin object is needed for circuit graphic rendering.
    """
    if output_format == Format.TEXT:
        return TextRenderer(n)
    if output_format == Format.LATEX:
        raise NotImplementedError("Latex format is not supported for circuit rendering")

    assert skin is not None, "A skin must be selected for circuit graphical rendering"
    if output_format == Format.HTML:
        canvas = SvgCanvas(**opts)
    else:
        canvas = MplotCanvas(**opts)
    return CanvasRenderer(n, canvas, skin)
