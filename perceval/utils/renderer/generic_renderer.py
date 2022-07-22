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
from abc import ABC, abstractmethod
from typing import List, Union, Literal, Tuple


class Canvas(ABC):
    def __init__(self, inverse_Y=False, **opts):
        self._position = []
        self._minx = None
        self._miny = None
        self._maxx = None
        self._maxy = None
        self._drawn = False
        self._offset_x = 0
        self._offset_y = 0
        if opts is None:
            self._opts = {}
        else:
            self._opts = opts
        self._inverse_Y = inverse_Y and -1 or 1

    def set_offset(self, v: Tuple[float, float], width: float, height: float):
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

    def add_mline(self,
                  points: List[float],
                  stroke: str = "black",
                  stroke_width: float = 1,
                  stroke_linejoin: str = "miter",
                  stroke_dasharray=None):
        """Draw a multi-line

        :param points:
        :param stroke:
        :param stroke_width:
        :return:
        """
        assert not self._drawn, "calling add_mline on drawn canvas"
        norm_points = []
        for x, y in [points[n:n+2] for n in range(0, len(points), 2)]:
            self.position = (x, y)
            norm_points += [self.position[0], self._inverse_Y * self.position[1]]
        return norm_points

    def add_polygon(self,
                    points: List[float],
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
                 points: Tuple[float, float],
                 width: float,
                 height: float,
                 **args):
        self.add_polygon([points[0], points[1],
                          points[0]+width, points[1],
                          points[0]+width, points[1]+height,
                          points[0], points[1]+height],
                         **args)

    def add_mpath(self,
                  points: List[Union[float, str]],
                  stroke: str = "black",
                  stroke_width: float = 1,
                  fill: str = None,
                  stroke_linejoin: str = "miter",
                  stroke_dasharray=None):
        """Draw a path

        :param fill:
        :param points:
        :param stroke:
        :param stroke_width:
        :return:
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
                raise RuntimeError("Unsupported mpath operator: %s", points[0])
        return norm_points

    def add_circle(self,
                   points: Tuple[float, float],
                   r: float,
                   stroke: str = "black",
                   stroke_width: float = 1,
                   fill: str = None,
                   stroke_dasharray = None):
        self.position = (points[0] + r, points[1] + r)
        self.position = (points[0] - r, points[1] - r)
        self.position = points
        return (self.position[0], self._inverse_Y * self.position[1])

    def add_text(self, points: Tuple[float, float],
                 text: str, size: float, ta: Literal["left", "middle", "right"] = "left",
                 fontstyle: Literal["normal", "bold", "italic"] = "normal"):
        self.position = points
        f_points = self.position
        if ta == "left":
            self.position = (points[0]+size*len(text), points[1]+size)
        elif ta == "right":
            self.position = (points[0]-size*len(text), points[1]+size)
        else:
            self.position = (points[0]-size*len(text)/2, points[1]+size)
            self.position = (points[0]+size*len(text)/2, points[1]+size)
        return (f_points[0], self._inverse_Y * f_points[1])

    def draw(self):
        assert not self._drawn, "calling draw on drawn canvas"
        self._drawn = True


class Renderer(ABC):
    @abstractmethod
    def new_canvas(self, **opts) -> Canvas:
        pass
