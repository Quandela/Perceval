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

from .canvas import Canvas

tikz_implemented_colors = {
    "red": True,
    "green": True,
    "blue": True,
    "cyan": True,
    "magenta": True,
    "yellow": True,
    "black": True,
    "gray": True,
    "white": True,
    "darkgray": True,
    "lightgray": True,
    "brown": True,
    "lime": True,
    "olive": True,
    "orange": True,
    "pink": True,
    "purple": True,
    "teal": True,
    "violet": True,
    "darkred": "\\definecolor{darkred}{rgb}{0.55, 0.0, 0.0}",
    "thistle": "\\definecolor{thistle}{rgb}{0.85, 0.75, 0.85}",
}


class LatexCanvas(Canvas):
    def __init__(self, **opts):
        super().__init__(**opts)
        self._header = [
            "\\documentclass{standalone}",
            "\\usepackage{tikz}",
        ]
        self._prescript = [
            "\\begin{document}",
            "\\begin{tikzpicture}[scale=1.5,x=1pt,y=1pt]",
        ]
        self._draws = []
        self._postscrip = [
            "\\end{tikzpicture}",
            "\\end{document}",
        ]
        self._color_check_list = tikz_implemented_colors.copy()

    def _check_color_is_implemented(self, color):
        if self._color_check_list[color] is None:
            raise NotImplementedError(f"Color {color} is not defined")
        elif self._color_check_list[color] is not True:
            self._header.append(self._color_check_list[color])
            self._color_check_list[color] = True

    def add_mline(
        self,
        points,
        stroke="black",
        stroke_width=1,
        stroke_linejoin="miter",
        stroke_dasharray=None,
    ):
        points = super().add_mline(points, stroke, stroke_width)
        self._check_color_is_implemented(stroke)

        nodes = []
        for idx in range(0, len(points), 2):
            nodes.append(f"({points[idx]},{-points[idx+1]})")
        self._draws.append(
            f"\draw[color={stroke},line width={stroke_width}] "
            + " -- ".join(nodes)
            + ";"
        )

    def add_polygon(
        self,
        points,
        stroke="black",
        stroke_width=1,
        fill=None,
        stroke_linejoin="miter",
        stroke_dasharray=None,
    ):
        points = super().add_polygon(points, stroke, stroke_width, fill)
        self._check_color_is_implemented(stroke)
        if fill is None:
            fill = "none"
        else:
            self._check_color_is_implemented(fill)

        nodes = []
        for idx in range(0, len(points), 2):
            nodes.append(f"({points[idx]},{-points[idx+1]})")
        self._draws.append(
            f"\draw[color={stroke},line width={stroke_width},fill={fill}] "
            + " -- ".join(nodes)
            + " -- cycle;"
        )

    def add_mpath(
        self,
        points,
        stroke="black",
        stroke_width=1,
        fill=None,
        stroke_linejoin="miter",
        stroke_dasharray=None,
    ):
        points = super().add_mpath(points, stroke, stroke_width, fill)
        self._check_color_is_implemented(stroke)
        if fill is None:
            fill = "none"
        else:
            self._check_color_is_implemented(fill)

        pathstr = f"\draw[color={stroke},line width={stroke_width},line join={stroke_linejoin},fill={fill}]"
        idx = 0
        x_pos, y_pos = 0, 0
        while idx < len(points):
            if points[idx] == "M":
                x_pos, y_pos = points[idx + 1 : idx + 3]
                idx += 2
            elif points[idx] == "L":
                x_end, y_end = points[idx + 1 : idx + 3]
                pathstr += f" ({x_pos},{-y_pos}) -- ({x_end},{-y_end})"
                x_pos, y_pos = x_end, y_end
                idx += 2
            elif points[idx] == "S":
                x_ctl_1, y_ctl_1 = x_ctl_2, y_ctl_2
                x_ctl_2, y_ctl_2, x_end, y_end = points[idx + 1 : idx + 5]
                pathstr += f" ({x_pos},{-y_pos}) .. controls ({x_ctl_1},{-y_ctl_1}) and ({x_ctl_2},{-y_ctl_2}) .. ({x_end},{-y_end})"
                x_pos, y_pos = x_end, y_end
                idx += 4
            elif points[idx] == "C":
                x_ctl_1, y_ctl_1, x_ctl_2, y_ctl_2, x_end, y_end = points[
                    idx + 1 : idx + 7
                ]
                pathstr += f" ({x_pos},{-y_pos}) .. controls ({x_ctl_1},{-y_ctl_1}) and ({x_ctl_2},{-y_ctl_2}) .. ({x_end},{-y_end})"
                x_pos, y_pos = x_end, y_end
                idx += 6
            idx += 1

        self._draws.append(pathstr + ";")

    def add_circle(
        self,
        points,
        r,
        stroke="black",
        stroke_width=1,
        fill=None,
        stroke_dasharray=None,
    ):
        points = super().add_circle(points, r, stroke, stroke_width, fill)
        if fill is None:
            fill = "none"

        self._draws.append(
            f"\draw[color={stroke},line width={stroke_width},line join={stroke_linejoin},fill={fill}] ({points[0]},{points[1]}) circle[radius={r}];"
        )

    def add_text(self, points, text, size, ta="left", fontstyle="normal"):
        if ta == "middle":
            ta = "mid"
        elif ta == "left":
            ta = "west"
        elif ta == "right":
            ta = "east"

        points = super().add_text(points, text, size, ta)

        if fontstyle == "normal":
            self._draws.append(
                f"\\node[anchor={ta},font = {{\\fontsize{{{size}pt}}{{0}}}}] at ({points[0]},{-points[1]}) {{{text}}};"
            )
        elif fontstyle == "italic":
            self._draws.append(
                f"\\node[align={ta},font = {{\\fontsize{{{size}pt}}{{0}}\\itshape}}] at ({points[0]},{-points[1]}) {{{text}}};"
            )
        elif fontstyle == "bold":
            self._draws.append(
                f"\\node[align={ta},font = {{\\fontsize{{{size}pt}}{{0}}\\bfseries}}] at ({points[0]},{-points[1]}) {{{text}}};"
            )
        else:
            raise NotImplementedError(f"Font style {fontstyle} not implemented")

    def draw(self):
        super().draw()
        return "\n".join(self._header + self._prescript + self._draws + self._postscrip)
