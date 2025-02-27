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
from __future__ import annotations
from abc import ABC


class Canvas(ABC):
    def __init__(self, inverse_Y: bool = False, **opts):
        self._position = []
        self._minx = None
        self._miny = None
        self._maxx = None
        self._maxy = None
        self._drawn = False
        self._offset_x = 0
        self._offset_y = 0
        self._opts = opts
        self._inverse_Y = -1 if inverse_Y else 1
        self._background_color = None

    def set_offset(self, v: tuple[float, float], width: float, height: float):
        self._offset_x = v[0]
        self._offset_y = v[1]
        self.position = (0, 0)
        self.position = (width, height)

    @property
    def position(self):
        return self._position

    @property
    def relative_position(self):
        return self._position[0]-self._offset_x, self._position[1]-self._offset_y

    @position.setter
    def position(self, v):
        (x, y) = (v[0]+self._offset_x, v[1]+self._offset_y)
        if self._minx is None or x < self._minx:
            self._minx = x
        if self._miny is None or y < self._miny:
            self._miny = y
        if self._maxx is None or x > self._maxx:
            self._maxx = x
        if self._maxy is None or y > self._maxy:
            self._maxy = y
        self._position = (x, y)

    def height(self):
        return self._maxy - self._miny

    def width(self):
        return self._maxx - self._minx

    def add_mline(self,
                  points: list[float],
                  stroke: str = "black",
                  stroke_width: float = 1,
                  stroke_linejoin: str = "miter",
                  stroke_dasharray=None):
        """Draw a multi-line

        :param points: list of point 2D coordinates
        :param stroke: Stroke color
        :param stroke_width: Width of the drawn multi-line
        :param stroke_linejoin: Shape to join two segments of the multi-line
        :param stroke_dasharray: Dash pattern of the multi-line
        """
        assert not self._drawn, "calling add_mline on drawn canvas"
        norm_points = []
        for x, y in [points[n:n+2] for n in range(0, len(points), 2)]:
            self.position = (x, y)
            norm_points += [self.position[0], self._inverse_Y * self.position[1]]
        return norm_points

    def add_polygon(self,
                    points: list[float],
                    stroke: str = "black",
                    stroke_width: float = 1,
                    fill: str = None,
                    stroke_linejoin: str = "miter",
                    stroke_dasharray=None,
                    inverse=True):
        """Draw a polygon

        :param fill:
        :param points:
        :param stroke:
        :param stroke_width:
        :return:
        """
        assert not self._drawn, "calling add_polygon on drawn canvas"
        norm_points = []
        for x, y in [points[n:n+2] for n in range(0, len(points), 2)]:
            self.position = (x, y)
            norm_points += [self.position[0], self._inverse_Y * self.position[1]]
        return norm_points

    def add_rect(self,
                 points: tuple[float, float],
                 width: float,
                 height: float,
                 **args):
        self.add_polygon([points[0], points[1],
                          points[0]+width, points[1],
                          points[0]+width, points[1]+height,
                          points[0], points[1]+height],
                         **args)

    def add_mpath(self,
                  points: list[float | str] | tuple[float | str, ...],
                  stroke: str = "black",
                  stroke_width: float = 1,
                  fill: str = None,
                  stroke_linejoin: str = "miter",
                  stroke_dasharray=None):
        """Draw a path

        :param points: list of point 2D coordinates
        :param stroke: Stroke color
        :param stroke_width: Width of the drawn multi-line
        :param fill: Fill color
        :param stroke_linejoin: Shape to join two segments of the multi-line
        :param stroke_dasharray: Dash pattern of the multi-line
        """
        assert not self._drawn, "calling add_mpath on drawn canvas"
        norm_points = []
        r_position_start = None
        while points:
            if points[0] == "z":
                self.position = r_position_start
                norm_points += ["L", self.position[0], self._inverse_Y * self.position[1]]
                points = points[1:]
            elif points[0] == "M":
                self.position = points[1:3]
                if r_position_start is None:
                    r_position_start = self.relative_position
                norm_points += ["M", self.position[0], self._inverse_Y * self.position[1]]
                points = points[3:]
            elif points[0] == "m":
                self.position = (self.relative_position[0]+points[1], self.relative_position[1]+points[2])
                norm_points += ["M", self.position[0], self._inverse_Y * self.position[1]]
                points = points[3:]
            elif points[0] == "L":
                self.position = points[1:3]
                norm_points += ["L", self.position[0], self._inverse_Y * self.position[1]]
                points = points[3:]
            elif points[0] == "l":
                self.position = (self.relative_position[0]+points[1], self.relative_position[1]+points[2])
                norm_points += ["L", self.position[0], self._inverse_Y * self.position[1]]
                points = points[3:]
            elif points[0] == "C":
                self.position = points[1:3]
                norm_points += ["C", self.position[0], self._inverse_Y * self.position[1]]
                self.position = points[3:5]
                norm_points += [self.position[0], self._inverse_Y * self.position[1]]
                self.position = points[5:7]
                norm_points += [self.position[0], self._inverse_Y * self.position[1]]
                points = points[7:]
            elif points[0] == "c":
                position_begin = self.relative_position
                self.position = (position_begin[0]+points[1], position_begin[1]+points[2])
                norm_points += ["C", self.position[0], self._inverse_Y * self.position[1]]
                self.position = (position_begin[0]+points[3], position_begin[1]+points[4])
                norm_points += [self.position[0], self._inverse_Y * self.position[1]]
                self.position = (position_begin[0]+points[5], position_begin[1]+points[6])
                norm_points += [self.position[0], self._inverse_Y * self.position[1]]
                points = points[7:]
            elif points[0] == "S":
                self.position = points[1:3]
                norm_points += ["S", self.position[0], self._inverse_Y * self.position[1]]
                self.position = points[3:5]
                norm_points += [self.position[0], self._inverse_Y * self.position[1]]
                points = points[5:]
            elif points[0] == "s":
                position_begin = self.relative_position
                self.position = (position_begin[0]+points[1], position_begin[1]+points[2])
                norm_points += ["S", self.position[0], self._inverse_Y * self.position[1]]
                self.position = (position_begin[0]+points[3], position_begin[1]+points[4])
                norm_points += [self.position[0], self._inverse_Y * self.position[1]]
                points = points[5:]
            elif points[0] == "H":
                self.position = (points[1], self.position[1])
                norm_points += ["L", self.position[0], self._inverse_Y * self.position[1]]
                points = points[2:]
            elif points[0] == "h":
                self.position = (points[1] + self.relative_position[0], self.relative_position[1])
                norm_points += ["L", self.position[0], self._inverse_Y * self.position[1]]
                points = points[2:]
            elif points[0] == "V":
                self.position = (self.position[0], points[1])
                norm_points += ["L", self.position[0], self._inverse_Y * self.position[1]]
                points = points[2:]
            elif points[0] == "v":
                self.position = (self.relative_position[0], points[1] + self.relative_position[1])
                norm_points += ["L", self.position[0], self._inverse_Y * self.position[1]]
                points = points[2:]
            else:
                raise RuntimeError(f"Unsupported mpath operator: {points[0]}")
        return norm_points

    def add_circle(self,
                   points: tuple[float, float],
                   r: float,
                   stroke: str = "black",
                   stroke_width: float = 1,
                   fill: str = None,
                   stroke_dasharray = None):
        """
        Draw a circle

        :param r: Radius
        :param points: list of point 2D coordinates
        :param stroke: Stroke color
        :param stroke_width: Width of the drawn circle
        :param fill: Fill color
        :param stroke_dasharray: Dash pattern of the circle
        """
        self.position = (points[0] + r, points[1] + r)
        self.position = (points[0] - r, points[1] - r)
        self.position = points
        return self.position[0], self._inverse_Y * self.position[1]

    def add_text(self, points: tuple[float, float],
                 text: str, size: float,
                 ta: str = "left",  # Literal["left", "middle", "right"]
                 fontstyle: str = "normal"  # Literal["normal", "bold", "italic"]
                 ):
        self.position = points
        f_points = self.position
        if ta == "left":
            self.position = (points[0]+size*len(text), points[1]+size)
        elif ta == "right":
            self.position = (points[0]-size*len(text), points[1]+size)
        else:
            self.position = (points[0]-size*len(text)/2, points[1]+size)
            self.position = (points[0]+size*len(text)/2, points[1]+size)
        return f_points[0], self._inverse_Y * f_points[1]

    def normalize_text(self, text: str, size: float, points: tuple[float, float], max_size: int, shift_factor: float) -> tuple[str, int, tuple[float, float]]:
        """
        If the text does not fit in max_size, reduce the font size or split text over several lines.
        :param text: the text to check
        :param size: the font size of the text
        :param points: the position on the canvas. The text can be moved vertically when lines are added.
        :param max_size: the maximum size that the text can take horizontally
        :param shift_factor: to move the text vertically when adding lines
        :return: Modified text, size and points.
        """
        if max_size is not None and size * len(text) > max_size:
            font_size_factor = 1.5
            new_size = int(max_size / len(text) * font_size_factor)
            if new_size < size / 2:
                size = size / 2
                max_len = max_size/size * font_size_factor
                # try to split text on spaces instead of reducing size
                pos_x, pos_y = points
                remaining = text
                new_text = ''
                min_index = int(max_len * 0.8)
                max_index = int(max_len * 1.2)
                while size * len(remaining) > max_size:
                    index = int(max_len)
                    if ' ' in remaining[min_index:max_index]:
                        index = remaining[min_index:max_index].index(' ') + min_index
                    new_text += remaining[:index] + '\n'
                    remaining = remaining[index:]
                    pos_y += shift_factor*size
                new_text += remaining
                text = new_text
                points = (pos_x, pos_y)
            else:
                size = new_size
        return text, size, points

    def add_shape(self, shape_fn, circuit, mode_style):
        shape_fn(circuit, self, mode_style)

    def set_background_color(self, background_color):
        """
        The canvas is not expected to change its background color in response
        to this, but a drawable element can retrieve this property to know
        whether it is drawn on a white or a colored surface.
        """
        self._background_color = background_color

    @property
    def background_color(self):
        return self._background_color

    def draw(self):
        assert not self._drawn, "calling draw on drawn canvas"
        self._drawn = True
