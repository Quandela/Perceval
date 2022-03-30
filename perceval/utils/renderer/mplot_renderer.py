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
    def __init__(self):
        super().__init__()
        self._fig, self._ax = plt.subplots()
        self._patches = []

    def add_mline(self, points, stroke="black", stroke_width=1, stroke_linejoin="miter",
                  stroke_dasharray=None, only_svg=False):
        if only_svg:
            return
        mpath = ["M", points[0], points[1]]
        for n in range(2, len(points), 2):
            mpath += ["L", points[n], points[n+1]]
        self.add_mpath(mpath, stroke=stroke, stroke_width=stroke_width, stroke_linejoin=stroke_linejoin)

    def add_polygon(self, points, stroke="black", stroke_width=1, fill=None, stroke_linejoin="miter",
                    stroke_dasharray=None, only_svg=False):
        if only_svg:
            return
        points = super().add_polygon(points, stroke, stroke_width, fill)
        self._patches.append(
            mpatches.Polygon([(points[n], -points[n+1]) for n in range(0, len(points), 2)],
                             fill=fill is not None, color=fill,
                             ec=stroke, linewidth=stroke_width, joinstyle=stroke_linejoin))

    def add_mpath(self, points, stroke="black", stroke_width=1, fill=None, stroke_linejoin="miter",
                  stroke_dasharray=None, only_svg=False):
        if only_svg:
            return
        points = super().add_mpath(points, stroke, stroke_width, fill)
        path_data = []
        while points:
            if points[0] == "M":
                path_data.append((mpath.Path.MOVETO, [points[1], -points[2]]))
                points = points[3:]
            elif points[0] == "L":
                path_data.append((mpath.Path.LINETO, [points[1], -points[2]]))
                points = points[3:]
            elif points[0] == "C":
                path_data.append((mpath.Path.CURVE4, [points[1], -points[2]]))
                path_data.append((mpath.Path.CURVE4, [points[3], -points[4]]))
                path_data.append((mpath.Path.CURVE4, [points[5], -points[6]]))
                points = points[7:]
            elif points[0] == "S":
                path_data.append((mpath.Path.CURVE3, [points[1], -points[2]]))
                path_data.append((mpath.Path.CURVE3, [points[3], -points[4]]))
                points = points[5:]
        codes, vertices = zip(*path_data)
        path = mpath.Path(vertices, codes)
        self._patches.append(mpatches.PathPatch(path,
                                                fill=fill is not None, color=fill,
                                                ec=stroke, linewidth=stroke_width,
                                                joinstyle=stroke_linejoin))

    def add_circle(self, points, r, stroke="black", stroke_width=1, fill=None,
                   stroke_dasharray=None, only_svg=False):
        if only_svg:
            return
        points = super().add_circle(points, r, stroke, stroke_width, fill)
        self._patches.append(mpatches.Circle((points[0], -points[1]), r,
                                             fill=fill is not None, color=fill,
                                             ec=stroke, linewidth=stroke_width))

    def add_text(self, points, text, size, ta="left", only_svg=False):
        if only_svg:
            return
        points = super().add_text(points, text, size, ta)
        if ta == "middle":
            ta = "center"
        plt.text(points[0], -points[1], text, ha=ta, size=size*3)

    def draw(self):
        super().draw()
        collection = PatchCollection(self._patches, match_original=True)
        self._ax.add_collection(collection)

        plt.axis('equal')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        return str(self)


class MplotRenderer(Renderer):
    def new_canvas(self) -> Canvas:
        assert plt is not None, "matplotlib is not installed"
        return MplotCanvas()
