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

from .generic_renderer import Renderer, Canvas


class SVGCanvas(Canvas):
    def __init__(self, **opts):
        super().__init__(**opts)
        self._canvas = []

    def add_mline(self, points, stroke="black", stroke_width=1, stroke_linejoin="miter",
                  stroke_dasharray=None):
        points = super().add_mline(points, stroke, stroke_width)
        self._canvas.append('<polyline points="%s" fill="transparent"'
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

    def add_text(self, points, text, size, ta="start"):
        if ta == "right":
            ta = "end"
        elif ta == "left":
            ta = "start"
        points = super().add_text(points, text, size, ta)
        self._canvas.append('<text x="%f" y="%f" font-size="%f" text-anchor="%s">%s</text>' % (
            points[0], points[1], size, ta, text
        ))

    def draw(self):
        super().draw()
        return "<svg width='%f' height='%f' viewBox='%f %f %f %f'>%s</svg>" % (
            (self._maxx-self._minx), (self._maxy-self._miny),
            self._minx, self._miny, self._maxx, self._maxy,
            "\n".join(self._canvas)
        )


class SVGRenderer(Renderer):
    def new_canvas(self, **opts) -> Canvas:
        return SVGCanvas(**opts)
