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

try:
    import matplotlib.pyplot as plt
    import matplotlib.path as mpath
    import matplotlib.patches as mpatches
    from matplotlib.collections import PatchCollection
except ModuleNotFoundError:
    plt = None
    mpath = None
    mpatches = None
    PatchCollection = None


class MplotCanvas(Canvas):
    def __init__(self, **opts):
        super().__init__(**opts, inverse_Y=True)
        self._fig, self._ax = plt.subplots()
        if "total_width" in opts:
            self._fig.set_figwidth((opts["total_width"]+1)*2)
            self._fig.set_figheight((opts["total_height"]+1)*2)
        self._patches = []

    def add_mline(self, points, stroke="black", stroke_width=1, stroke_linejoin="miter",
                  stroke_dasharray=None):
        mpath = ["M", points[0], points[1]]
        for n in range(2, len(points), 2):
            mpath += ["L", points[n], points[n+1]]
        self.add_mpath(mpath, stroke=stroke, stroke_width=stroke_width, stroke_linejoin=stroke_linejoin)

    def add_polygon(self, points, stroke="black", stroke_width=1, fill=None, stroke_linejoin="miter",
                    stroke_dasharray=None):
        points = super().add_polygon(points, stroke, stroke_width, fill)
        self._patches.append(
            mpatches.Polygon([(points[n], points[n+1]) for n in range(0, len(points), 2)],
                             fill=fill is not None, color=fill,
                             ec=stroke, linewidth=stroke_width, joinstyle=stroke_linejoin))

    def add_mpath(self, points, stroke="black", stroke_width=1, fill=None, stroke_linejoin="miter",
                  stroke_dasharray=None):
        points = super().add_mpath(points, stroke, stroke_width, fill)
        path_data = []
        while points:
            if points[0] == "M":
                path_data.append((mpath.Path.MOVETO, [points[1], points[2]]))
                points = points[3:]
            elif points[0] == "L":
                path_data.append((mpath.Path.LINETO, [points[1], points[2]]))
                points = points[3:]
            elif points[0] == "C":
                path_data.append((mpath.Path.CURVE4, [points[1], points[2]]))
                path_data.append((mpath.Path.CURVE4, [points[3], points[4]]))
                path_data.append((mpath.Path.CURVE4, [points[5], points[6]]))
                points = points[7:]
            elif points[0] == "S":
                path_data.append((mpath.Path.CURVE3, [points[1], points[2]]))
                path_data.append((mpath.Path.CURVE3, [points[3], points[4]]))
                points = points[5:]
        codes, vertices = zip(*path_data)
        path = mpath.Path(vertices, codes)
        self._patches.append(mpatches.PathPatch(path,
                                                fill=fill is not None, color=fill,
                                                ec=stroke, linewidth=stroke_width,
                                                joinstyle=stroke_linejoin))

    def add_circle(self, points, r, stroke="black", stroke_width=1, fill=None,
                   stroke_dasharray=None):
        points = super().add_circle(points, r, stroke, stroke_width, fill)
        self._patches.append(mpatches.Circle((points[0], points[1]), r,
                                             fill=fill is not None, color=fill,
                                             ec=stroke, linewidth=stroke_width))

    def add_text(self, points, text, size, ta="left", fontstyle="normal"):
        points = super().add_text(points, text, size, ta)
        if ta == "middle":
            ta = "center"
        kwargs = {
            'ha': ta,
            'size': size*3
        }
        if fontstyle == "italic":
            kwargs['fontstyle'] = fontstyle
        elif fontstyle == "bold":
            kwargs['fontweight'] = fontstyle
        plt.text(points[0], points[1], text, **kwargs)

    def draw(self):
        super().draw()
        collection = PatchCollection(self._patches, match_original=True)
        self._ax.add_collection(collection)

        plt.axis('equal')
        plt.axis('off')
        plt.tight_layout()
        if "mplot_savefig" in self._opts:
            plt.savefig(self._opts["mplot_savefig"])
        if "mplot_noshow" not in self._opts or not self._opts["mplot_noshow"]:
            plt.show()
        plt.close(self._fig)
        return self


class MplotRenderer(Renderer):
    def new_canvas(self, **opts) -> Canvas:
        assert plt is not None, "matplotlib is not installed"
        return MplotCanvas(**opts)
