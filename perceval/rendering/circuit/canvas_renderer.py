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

from copy import copy

from .renderer_interface import ICircuitRenderer
from ..canvas import Canvas
from perceval.components import ACircuit, APort, PortLocation, PERM, IDetector, Herald
from perceval.utils import ModeType, BasicState


class _PortPos:
    def __init__(self, x, y):
        self.x = x
        self.y = y


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

        self._in_port_pos = []
        self._out_port_pos = []
        for i in range(nsize):
            self._in_port_pos.append(_PortPos(0, i))
        for i in range(nsize):
            self._out_port_pos.append(_PortPos(0, i))

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

    def get_circuit_size(self, circuit: ACircuit, recursive: bool = False) -> tuple[int, int]:
        return self._skin.get_size(circuit, recursive)

    def display_input_photons(self, input_pos: BasicState) -> None:
        """
        Display half-cup showing the number of expected photons at the beginning of any mode
        """
        for k in range(input_pos.m):
            if self._mode_style[k] != ModeType.HERALD:
                self._canvas.set_offset(
                    (-CanvasRenderer.AFFIX_ALL_SIZE * 1.5, CanvasRenderer.AFFIX_ALL_SIZE * 2 * k), 0, 0)
                self._canvas.add_mline([
                    CanvasRenderer.AFFIX_ALL_SIZE, CanvasRenderer.AFFIX_ALL_SIZE,
                    CanvasRenderer.SCALE - 2, CanvasRenderer.AFFIX_ALL_SIZE],
                    **self._skin.style[ModeType.PHOTONIC])
                h = Herald(input_pos[k])
                self._canvas.add_shape(self._skin.get_shape(h, PortLocation.INPUT), h, None)

    def add_mode_index(self):
        self._canvas.set_offset(
            (CanvasRenderer.AFFIX_ALL_SIZE + max(self._chart) * CanvasRenderer.SCALE, 0),
            CanvasRenderer.AFFIX_ALL_SIZE,
            CanvasRenderer.SCALE * (self._nsize + 1))
        for k in range(self._nsize):
            if self._mode_style[k] != ModeType.HERALD:
                self._canvas.add_text(
                    (CanvasRenderer.AFFIX_ALL_SIZE, CanvasRenderer.SCALE * (k + 0.5) + 3),
                    str(k),
                    self._n_font_size,
                    ta="right")

        self._canvas.set_offset(
            (0, 0),
            CanvasRenderer.AFFIX_ALL_SIZE,
            CanvasRenderer.SCALE * (self._nsize + 1))
        for k in range(self._nsize):
            if self._mode_style[k] != ModeType.HERALD:
                self._canvas.add_text(
                    (
                        0,
                        CanvasRenderer.SCALE / 2 + 3 + CanvasRenderer.SCALE * k
                    ),
                    str(k),
                    self._n_font_size,
                    ta="left")

    def add_out_port(self, n_mode: int, port: APort):
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
        self._canvas.add_shape(self._skin.get_shape(port, PortLocation.OUTPUT), port, None)

    def add_in_port(self, n_mode: int, port: APort):
        h_pos = self._in_port_pos[n_mode].x * CanvasRenderer.SCALE
        v_pos = self._in_port_pos[n_mode].y * CanvasRenderer.SCALE
        self._canvas.set_offset(
            (h_pos, v_pos),
            CanvasRenderer.AFFIX_ALL_SIZE,
            CanvasRenderer.SCALE)
        self._canvas.add_shape(self._skin.get_shape(port, PortLocation.INPUT), port, None)

    def add_detectors(self, detector_list: list) -> None:
        max_pos = max(self._chart[0:self._nsize])
        for i, det in enumerate(detector_list):
            if det is None or self._mode_style[i] != ModeType.PHOTONIC:
                continue
            self._canvas.set_offset(
                (
                    CanvasRenderer.AFFIX_ALL_SIZE + CanvasRenderer.SCALE * (max_pos + .5),
                    CanvasRenderer.SCALE * i
                ),
                CanvasRenderer.AFFIX_ALL_SIZE,
                CanvasRenderer.SCALE)
            self._canvas.add_shape(self._skin.get_shape(det), det, [None])

    def open_subblock(self, lines: tuple[int, ...], name: str, size: tuple[int, int], color=None):
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
            self.extend_pos(start, end, margins[0])

    def close_subblock(self, lines: tuple[int, ...]):
        start = lines[0]
        end = lines[-1]
        right_margins = self._current_subblock_info.get('margins', (0, 0))[1]
        # Extend lines on the right side
        self.extend_pos(start, end, right_margins)

    def max_pos(self, start, end, _=None):
        return max(self._chart[start:end + 1])

    def extend_pos(self, start, end, margin: int = 0):
        maxpos = self.max_pos(start, end) + margin
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

    def _add_shape(self, lines, circuit, w, shape_fn=None):
        if shape_fn is None:
            shape_fn = self._skin.get_shape(circuit)
        start = lines[0]
        end = lines[-1]
        self.extend_pos(start, end)
        max_pos = self.max_pos(start, end)
        self._canvas.set_offset(
            (
                CanvasRenderer.AFFIX_ALL_SIZE + CanvasRenderer.SCALE * max_pos,
                CanvasRenderer.SCALE * start
            ),
            CanvasRenderer.SCALE * w,
            CanvasRenderer.SCALE * (end - start + 1))
        modes = self._mode_style[start:(end + 1)]
        self._canvas.add_shape(shape_fn, circuit, modes)

    def set_herald_info(self, info):
        self._herald_info = info

    def _update_mode_style(self, lines, circuit, w: int):
        if isinstance(circuit, IDetector):
            self._mode_style[lines[0]] = ModeType.CLASSICAL

        elif not isinstance(circuit, PERM):
            input_heralds = {}
            output_heralds = {}

            herald_info = self._herald_info if self._herald_info else {}
            if circuit in herald_info:
                output_heralds = herald_info[circuit].output_heralds
                input_heralds = herald_info[circuit].input_heralds

            # Position input and output heralds
            for in_mode, herald_in_mode in input_heralds.items():
                self._in_port_pos[herald_in_mode].y = lines[0] + in_mode
                self._in_port_pos[herald_in_mode].x = self._chart[lines[0] + in_mode]
                # Start drawing this mode in "photonic" style
                self._mode_style[lines[0] + in_mode] = ModeType.PHOTONIC
            for out_mode, herald_out_mode in output_heralds.items():
                self._out_port_pos[herald_out_mode].y = lines[0] + out_mode
                self._out_port_pos[herald_out_mode].x = self._chart[lines[0] + out_mode] + w
                # Stop drawing this mode (set it in "herald" style)
                self._mode_style[lines[0] + out_mode] = ModeType.HERALD

        else:  # Permutation case
            m0 = lines[0]
            out_modes = copy(self._mode_style)
            for m_input, m_output in enumerate(circuit.perm_vector):
                out_modes[m_output + lines[0]] = self._mode_style[m_input + m0]
            self._mode_style = out_modes

    def append_circuit(self, lines, circuit):
        w = self._skin.get_width(circuit)
        self._add_shape(lines, circuit, w)
        self._update_mode_style(lines, circuit, w)
        for i in range(lines[0], lines[-1] + 1):
            self._chart[i] += w

    def append_subcircuit(self, lines, circuit):
        w = self._skin.style_subcircuit['width']
        if w:
            self._add_shape(lines, circuit, w, self._skin.subcircuit_shape)
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

    def display_input_photons(self, input_pos) -> None:
        pass

    def add_mode_index(self) -> None:
        pass

    def add_out_port(self, m: int, port: APort) -> None:
        pass

    def add_in_port(self, m: int, content: str) -> None:
        pass

    def add_detectors(self, detector_list: list) -> None:
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

    def extend_pos(self, start: int, end: int, margin: int = 0):
        maxpos = self.max_pos(start, end) + margin
        for p in range(start, end + 1):
            self._chart[p] = maxpos

    def _add_shape(self, lines, circuit, w, shape_fn=None):
        self.extend_pos(lines[0], lines[-1])

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

    def append_circuit(self, lines, circuit):
        w = self._skin.get_width(circuit)
        if w:
            self._add_shape(lines, circuit, w)
            self._update_mode_style(lines, circuit, w)
            for i in range(lines[0], lines[-1] + 1):
                self._chart[i] += w

    def append_subcircuit(self, lines, circuit):
        w = self._skin.style_subcircuit['width']
        if w:
            self._add_shape(lines, circuit, w, self._skin.subcircuit_shape)
            self._update_mode_style(lines, circuit, w)
            for i in range(lines[0], lines[-1] + 1):
                self._chart[i] += w
