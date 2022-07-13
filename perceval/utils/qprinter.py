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

import math
from .renderer import SVGRenderer, MplotRenderer, Canvas
from .renderer.shapes import detector_shape, source_shape
from .utils import simple_float

"""
    Generic (abstract) class for drawing a LO schema - it covers different levels of representation:
    - core component
    - circuit
    - processor
"""


class TextPrinter:
    def __init__(self, nsize, hc=3):
        self._nsize = nsize
        self._hc = hc
        self._h = ['']*(hc*nsize+2)
        for k in range(nsize):
            self._h[hc*k+2] = "──"
        self.extend_pos(0, self._nsize-1)
        self._depth = [0]*nsize
        self._offset = 0

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

    def open_subblock(self, lines, name, area=None, internal=False):
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
        return None

    def append_subcircuit(self, lines, circuit, content, min_size=5):
        self.open_subblock(lines, circuit._name)
        self.extend_pos(lines[0], lines[-1], header=True, internal=True, char="░")
        self.close_subblock(lines)

    def append_circuit(self, lines, circuit, content, min_size=5):
        # opening the box
        start = lines[0]
        end = lines[-1]
        self.extend_pos(start, end)
        # put variables on the right number of lines
        content = circuit._name + (content and "\n"+content or "")
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
        if maxw < min_size:
            maxw = min_size
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
        self._set_offset(2)
        for k in range(self._nsize):
            self._h[self._hc*k + 2] = str(k) + ':' + self._h[self._hc*k + 2][2:]
            self._h[self._hc*k + 2] += ':' + str(k) + f" (depth {self._depth[k]})"

    def add_out_port(self, m, **opts):
        content = ''
        if 'content' in opts and opts['content']:
            content = opts['content']
        self._h[self._hc*m + 2] += f'[{content})'
        if 'name' in opts and opts['name']:
            self._h[self._hc*m + 3] += f"[{opts['name']}]"

    def add_in_port(self, m, **opts):
        content = ''
        if 'content' in opts and opts['content']:
            content = opts['content']
        shape_size = len(content) + 2
        name = ''
        if 'name' in opts and opts['name']:
            name = '[' + opts['name'] + ']'
        name_size = len(name)
        self._set_offset(max(shape_size, name_size))
        self._h[self._hc*m + 2] = f'({content}]' + '─'*(self._offset-shape_size) \
                                  + self._h[self._hc*m + 2][self._offset:]
        self._h[self._hc*m + 3] = name + ' '*(self._offset-name_size) + self._h[self._hc*m + 3][self._offset:]


class GraphicPrinter:
    affix_port_size = 15
    affix_all_size = 25

    def __init__(self, nsize, canvas: Canvas, stroke_style, compact_rendering=False):
        self._stroke_style = stroke_style
        self._nsize = nsize
        # first position available for row n
        self._chart = [0] * (nsize+1)
        self._canvas = canvas
        self._canvas.set_offset((0, 0),
                                GraphicPrinter.affix_all_size, 50 * (nsize + 1))
        self._n_font_size = min(12, max(6, self._nsize+1))
        for k in range(nsize):
            self._canvas.add_mpath(["M", GraphicPrinter.affix_all_size-GraphicPrinter.affix_port_size, 25 + 50 * k,
                                    "l", GraphicPrinter.affix_port_size, 0], **self._stroke_style)
        self._current_block_open_offset = None
        self._current_block_name = ""
        self._compact = compact_rendering

    def add_mode_index(self):
        for k in range(self._nsize):
            self._canvas.add_text((GraphicPrinter.affix_all_size, 25 + 50 * k), str(k), self._n_font_size, ta="right")

        self._canvas.set_offset((0, 0), GraphicPrinter.affix_all_size, 50 * (self._nsize + 1))
        for k in range(self._nsize):
            self._canvas.add_text((0, 25 + 50 * k), str(k), self._n_font_size, ta="left")

    def add_out_port(self, n_mode, **opts):
        max_pos = self.extend_pos(0, self._nsize - 1)
        self._canvas.set_offset((GraphicPrinter.affix_all_size + 50*max_pos, 50*n_mode),
                                GraphicPrinter.affix_all_size, 50*(n_mode + 1))
        detector_shape(self._canvas, **opts)

    def add_in_port(self, n_mode, **opts):
        self._canvas.set_offset((0, 50*n_mode),
                                GraphicPrinter.affix_all_size, 50*(n_mode + 1))
        source_shape(self._canvas, **opts)

    def open_subblock(self, lines, name, area=None, color=None):
        start = lines[0]
        end = lines[-1]
        self._current_block_open_offset = self.extend_pos(start, end)
        self._current_block_name = name
        if area is not None:
            self._canvas.set_offset((GraphicPrinter.affix_all_size + 50 * area[0], 50 * area[1]),
                                    50 * area[2], 50 * area[3])
            if color is None:
                color = "lightblue"
            self._canvas.add_rect((2,2), 50 * area[2]-4, 50 * area[3]-4, fill=color, stroke="none")

    def close_subblock(self, lines):
        start = lines[0]
        end = lines[-1]
        begpos = self._current_block_open_offset
        curpos = self.extend_pos(start, end)
        self._canvas.set_offset((GraphicPrinter.affix_all_size+50 * begpos, 50 * start),
                                50 * (curpos-begpos), 50 * (end - start + 1))
        self._canvas.add_rect((2, 2), 50 * (curpos-begpos)-4, 50 * (end - start + 1)-4,
                              stroke_dasharray="1,2")
        self._canvas.add_text((4, 50 * (end - start + 1)+5), self._current_block_name.upper(), 8)
        return (begpos, start, (curpos-begpos), (end-start+1))

    def max_pos(self, start, end, _):
        return max(self._chart[start:end+1])

    def extend_pos(self, start, end):
        maxpos = max(self._chart[start:end+1])
        for p in range(start, end+1):
            if self._chart[p] != maxpos:
                self._canvas.set_offset((GraphicPrinter.affix_all_size+self._chart[p]*50, p*50),
                                        (maxpos-self._chart[p])*50, 50)
                self._canvas.add_mline([0, 25, (maxpos-self._chart[p])*50, 25], **self._stroke_style)
            self._chart[p] = maxpos
        return maxpos

    def append_circuit(self, lines, circuit, content):
        # opening the box
        start = lines[0]
        end = lines[-1]
        max_pos = self.extend_pos(start, end)
        w = circuit.get_width(self._compact)
        self._canvas.set_offset((GraphicPrinter.affix_all_size+50*max_pos, 50*start), 50*w, 50*(end-start+1))
        circuit.shape(content, self._canvas, self._compact)
        for i in range(start, end+1):
            self._chart[i] += w

    def append_subcircuit(self, lines, circuit, content):
        if circuit.stroke_style:
            self._stroke_style = circuit.stroke_style
        # opening the box
        start = lines[0]
        end = lines[-1]
        max_pos = self.extend_pos(start, end)
        w = circuit.subcircuit_width
        self._canvas.set_offset((GraphicPrinter.affix_all_size+50*max_pos, 50*start), 50*w, 50*(end-start+1))
        #circuit.shape(content, self._canvas)
        circuit.subcircuit_shape(circuit._name, self._canvas)
        for i in range(start, end+1):
            self._chart[i] += w

    def close(self):
        max_pos = self.extend_pos(0, self._nsize-1)
        self._canvas.set_offset((GraphicPrinter.affix_all_size+50*max_pos, 0),
                                GraphicPrinter.affix_all_size, 50*(self._nsize+1))
        for k in range(self._nsize):
            self._canvas.add_mpath(["M", 0, 25 + 50 * k,
                                    "l", GraphicPrinter.affix_port_size, 0], **self._stroke_style)

    def draw(self):
        return self._canvas.draw()


def create_printer(n, output_format="text", stroke_style="", compact=False, **opts):
    if output_format == "text":
        return TextPrinter(n)
    elif output_format == "latex":
        raise NotImplementedError("latex format not yet supported")
    if output_format == "html" or output_format == "png":
        canvas = SVGRenderer().new_canvas(**opts)
    else:
        canvas = MplotRenderer().new_canvas(**opts)
    return GraphicPrinter(n, canvas, stroke_style, compact)


SPECIAL_OUTPUTS = {
    'PERM': '_╲ ╱\n_ ╳ \n_╱ ╲'
}


def format_parameters(params: dict, precision: float = 1e-6, nsimplify: bool = True, separator: str = '\n') -> str:
    """
    Prepares a string output from a dictionnary of paramaters.
    params: dictionnary where keys are the parameter names and values are the corresponding parameter value. Values can
            either be a string or a float.
            If a key is found in SPECIAL_OUTPUTS, the value is replaced by the hardcoded value.
    precision: Rounds a float value to the given precision
    nsimplify: Try to simplify numerical display in case of float value
    separator: String separator for the final join
    """
    output = []
    for key, value in params.items():
        if key in SPECIAL_OUTPUTS:
            output.append(SPECIAL_OUTPUTS[key])
            continue

        if not isinstance(value, str):
            _, value = simple_float(value, precision, nsimplify)
        output.append(f'{key}={value}')
    return separator.join(output)
