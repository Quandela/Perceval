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

from .renderer_interface import ICircuitRenderer
from perceval.components import ACircuit, Barrier, PERM, APort, Herald
from perceval.utils import format_parameters


class TextRenderer(ICircuitRenderer):
    _PERM_DESC = '_╲ ╱\n_ ╳ \n_╱ ╲'  # ASCII art representation of a permutation

    def __init__(self, nsize, hc=3, min_box_size=5):
        super().__init__(nsize)
        self._hc = hc
        self._h = [''] * (hc * nsize + 2)
        self._depth = [0] * nsize
        self._offset = 0
        self.min_box_size = min_box_size
        self._ext_char = " "
        self.extend_pos(0, self._nsize - 1)

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

    def extend_pos(self, start, end, internal=False, header=False):
        char = self._ext_char
        maxpos = self.max_pos(start, end, header)
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

    def append_subcircuit(self, lines, circuit):
        self.open_subblock(lines, circuit.name, None)
        self._ext_char = "░"
        self.extend_pos(lines[0], lines[-1], header=True, internal=True)
        self._ext_char = " "
        self.close_subblock(lines)

    def append_circuit(self, lines, circuit):
        # opening the box
        start = lines[0]
        end = lines[-1]
        self.extend_pos(start, end)

        if isinstance(circuit, Barrier):
            if circuit.visible:
                for k in range(start * self._hc + 1, (end + 1) * self._hc + 1):
                    if k % self._hc == 2:
                        self._h[k] += "──║──"
                    else:
                        self._h[k] += "  ║  "
                self.extend_pos(start, end)
            return

        # put variables on the right number of lines
        if isinstance(circuit, PERM):
            content = self._PERM_DESC
        elif isinstance(circuit, ACircuit):
            content = format_parameters(circuit.get_variables(), 1e-3, False)
        else:
            content = ""
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
        self.extend_pos(start, end, internal=True)
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

    def display_input_photons(self, input_pos) -> None:
        pass  # Don't display input photons in text mode

    def add_out_port(self, n_mode: int, port: APort):
        content = ''
        if isinstance(port, Herald):
            content = port.expected
        for i in range(port.m):
            self._h[self._hc * (n_mode + i) + 2] += f'[{content})'
            self._h[self._hc * (n_mode + i) + 3] += f"[{port.name}]"

    def add_in_port(self, n_mode: int, port: APort):
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

    def add_detectors(self, detector_list: list) -> None:
        pass  # Don't display detectors in text mode
