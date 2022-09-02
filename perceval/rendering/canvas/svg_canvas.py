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

from .canvas import Canvas
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings(
        action='ignore',
        category=RuntimeWarning)
    import drawSvg as draw


class SvgCanvas(Canvas):
    """
    This class relies on drawSvg 3rd party library.
    With it, it is possible to create dynamic svg graphics.
    """
    def __init__(self, **opts):
        super().__init__(**opts, inverse_Y=True)
        self._draws = []
        self._render_width = (opts["total_width"]+3)*50
        self._render_height = (opts["total_height"]+1)*50
        self._pixel_size = 1.25
        if 'render_size' in opts:
            self._pixel_size *= opts['render_size']
        if 'group' in opts:
            self._group = opts['group']

    def add_mline(self, points, stroke="black", stroke_width=1, stroke_linejoin="miter",
                  stroke_dasharray=None):
        points = super().add_mline(points, stroke, stroke_width)
        self._draws.append(draw.Lines(*points, stroke=stroke, stroke_width=stroke_width,
                                      fill="none", close=False))

    def add_polygon(self, points, stroke="black", stroke_width=1, fill=None, stroke_linejoin="miter",
                    stroke_dasharray=None):
        points = super().add_polygon(points, stroke, stroke_width, fill)
        if fill is None:
            fill = "none"
        self._draws.append(draw.Lines(*points, stroke=stroke, fill=fill, close=True,
                                      stroke_dasharray=stroke_dasharray,
                                      stroke_linejoin=stroke_linejoin))

    def add_mpath(self, points, stroke="black", stroke_width=1, fill=None, stroke_linejoin="miter",
                  stroke_dasharray=None):
        points = super().add_mpath(points, stroke, stroke_width, fill)
        if fill is None:
            fill = "none"
        p = draw.Path(stroke_width=stroke_width, stroke=stroke, stroke_linejoin=stroke_linejoin,
                      fill=fill)
        idx = 0
        while idx < len(points):
            if points[idx] == 'M':
                p.M(*points[idx+1:idx+3])
                idx += 2
            elif points[idx] == 'L':
                p.L(*points[idx + 1:idx + 3])
                idx += 2
            elif points[idx] == 'S':
                p.S(*points[idx + 1:idx + 5])
                idx += 4
            elif points[idx] == 'C':
                p.C(*points[idx+1:idx+7])
                idx += 6
            idx += 1
        self._draws.append(p)

    def add_circle(self, points, r, stroke="black", stroke_width=1, fill=None,
                   stroke_dasharray=None):
        points = super().add_circle(points, r, stroke, stroke_width, fill)
        if fill is None:
            fill = "none"
        self._draws.append(draw.Circle(points[0], points[1], r,
                                       stroke_width=stroke_width, fill=fill, stroke=stroke))

    def add_text(self, points, text, size, ta="start", fontstyle="normal"):
        if ta == "right":
            ta = "end"
        elif ta == "left":
            ta = "start"
        points = super().add_text(points, text, size, ta)
        opts = {'text_anchor': ta}
        if fontstyle == "italic":
            opts['font_style'] = "italic"
        elif fontstyle == "bold":
            opts['font_weight'] = "bold"
        self._draws.append(draw.Text(text, size, *points, **opts))

    def draw(self):
        super().draw()
        if hasattr(self, "_group"):
            d = draw.Group(x=self._group[0], y=self._group[1])
        else:
            d = draw.Drawing(self._render_width, self._render_height,
                             origin=(self._minx-25, -self._maxy))
        for dr in self._draws:
            d.append(dr)
        return d.setPixelScale(self._pixel_size)
