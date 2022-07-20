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

from __future__ import annotations

import sys
from .generic_renderer import Renderer, Canvas


class StandardSVGCanvas(Canvas):
    """
    This class is the original homemade SVG canvas
    """
    def __init__(self, **opts):
        super().__init__(**opts)
        self._canvas = []

    def add_mline(self, points, stroke="black", stroke_width=1, stroke_linejoin="miter",
                  stroke_dasharray=None):
        points = super().add_mline(points, stroke, stroke_width)
        self._canvas.append('<polyline points="%s" fill="transparent" '
                            'stroke="%s" stroke-width="%f" stroke-linejoin="%s" %s/>' % (
            " ".join([str(p) for p in points]),
            stroke,
            stroke_width,
            stroke_linejoin,
            stroke_dasharray is not None and "stroke-dasharray="+stroke_dasharray or ""
        ))

    def add_polygon(self, points, stroke="black", stroke_width=1, fill=None, stroke_linejoin="miter",
                    stroke_dasharray=None):
        points = super().add_polygon(points, stroke, stroke_width, fill)
        self._canvas.append('<polyline points="%s" fill="%s" stroke="%s" stroke-width="%f" stroke_linejoin="%s" %s/>' % (
            " ".join([str(p) for p in points] + [str(points[0]), str(points[1])]),
            fill is None and "none" or fill,
            stroke,
            stroke_width,
            stroke_linejoin,
            stroke_dasharray is not None and "stroke-dasharray='"+stroke_dasharray+"'" or ""
        ))

    def add_mpath(self, points, stroke="black", stroke_width=1, fill=None, stroke_linejoin="miter",
                  stroke_dasharray=None):
        points = super().add_mpath(points, stroke, stroke_width, fill)
        self._canvas.append('<path d="%s" fill="%s" stroke="%s" stroke-width="%f" stroke-linejoin="%s"/>' % (
            " ".join([str(p) for p in points]),
            fill is None and "none" or fill,
            stroke,
            stroke_width,
            stroke_linejoin
        ))

    def add_circle(self, points, r, stroke="black", stroke_width=1, fill=None,
                   stroke_dasharray=None):
        points = super().add_circle(points, r, stroke, stroke_width, fill)
        self._canvas.append('<circle cx="%f" cy="%f" r="%f" stroke-width="%f" fill="%s" stroke="%s"/>' % (
            points[0], points[1], r,
            stroke_width,
            fill is None and "none" or fill,
            stroke
        ))

    def add_text(self, points, text, size, ta="start", fontstyle="normal"):
        if ta == "right":
            ta = "end"
        elif ta == "left":
            ta = "start"
        points = super().add_text(points, text, size, ta)
        additional_style = ''
        if fontstyle == "italic":
            additional_style = 'font-style="italic"'
        elif fontstyle == "bold":
            additional_style = 'font-weight="bold"'
        self._canvas.append('<text x="%f" y="%f" font-size="%f" %s text-anchor="%s">%s</text>' % (
            points[0], points[1], size, additional_style, ta, text
        ))

    def draw(self):
        super().draw()
        return "<svg width='%f' height='%f' viewBox='%f %f %f %f'>%s</svg>" % (
            (self._maxx-self._minx), (self._maxy-self._miny),
            self._minx, self._miny, self._maxx, self._maxy,
            "\n".join(self._canvas)
        )


class DynamicSVGCanvas(Canvas):
    """
    This class relies on drawSvg 3rd party library.
    DrawSvg relies on libcairo 2 which is not provided for Windows.
    Thus, only a prior import of drawSvg will use this class instead of the StandardSVGCanvas transfering the 3rd party
    responsibility to the user.

    However, with drawSvg, it is possible to create dynamic svg graphics.
    """
    def __init__(self, render_size=None, **opts):
        super().__init__(**opts, inverse_Y=True)
        self._draws = []
        self._render_size = render_size
        if 'group' in opts:
            self._group = opts['group']

    def add_mline(self, points, stroke="black", stroke_width=1, stroke_linejoin="miter",
                  stroke_dasharray=None):
        import drawSvg as draw  # done at method level in order to avoid an ImportError on Windows OS
        points = super().add_mline(points, stroke, stroke_width)
        self._draws.append(draw.Lines(*points, stroke=stroke, stroke_width=stroke_width,
                                      fill="none", close=False))

    def add_polygon(self, points, stroke="black", stroke_width=1, fill=None, stroke_linejoin="miter",
                    stroke_dasharray=None):
        import drawSvg as draw  # done at method level in order to avoid an ImportError on Windows OS
        points = super().add_polygon(points, stroke, stroke_width, fill)
        if fill is None:
            fill = "none"
        self._draws.append(draw.Lines(*points, stroke=stroke, fill=fill, close=True,
                                      stroke_dasharray=stroke_dasharray,
                                      stroke_linejoin=stroke_linejoin))

    def add_mpath(self, points, stroke="black", stroke_width=1, fill=None, stroke_linejoin="miter",
                  stroke_dasharray=None):
        import drawSvg as draw  # done at method level in order to avoid an ImportError on Windows OS
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
        import drawSvg as draw  # done at method level in order to avoid an ImportError on Windows OS
        points = super().add_circle(points, r, stroke, stroke_width, fill)
        if fill is None:
            fill = "none"
        self._draws.append(draw.Circle(points[0], points[1], r,
                                       stroke_width=stroke_width, fill=fill, stroke=stroke))

    def add_text(self, points, text, size, ta="start", fontstyle="normal"):
        import drawSvg as draw  # done at method level in order to avoid an ImportError on Windows OS
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
        import drawSvg as draw  # done at method level in order to avoid an ImportError on Windows OS
        super().draw()
        if hasattr(self, "_group"):
            d = draw.Group(x=self._group[0], y=self._group[1])
        else:
            d = draw.Drawing(self._maxx-self._miny, self._maxy-self._miny,
                             origin=(self._minx, -self._maxy))
        for dr in self._draws:
            d.append(dr)
        if self._render_size is not None:
            return d.setPixelScale(self._render_size)
        else:
            return d


class SVGRenderer(Renderer):
    def new_canvas(self, **opts) -> Canvas:
        # DynamicSVGCanvas is used only if drawSvg was imported beforehand
        if 'drawSvg' in sys.modules:
            return DynamicSVGCanvas(**opts)
        return StandardSVGCanvas(**opts)
