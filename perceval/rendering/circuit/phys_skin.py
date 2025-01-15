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
from multipledispatch import dispatch

from perceval.components import (AComponent, AFFConfigurator, Circuit,
                                 Port, Herald, PortLocation, IDetector, DetectionType,
                                 unitary_components as cp,
                                 non_unitary_components as nu)
from ._canvas_shapes import ShapeFactory
from .abstract_skin import ASkin, ModeType
from .skin_common import bs_convention_color


class PhysSkin(ASkin):
    _STROKE_WIDTH = 3

    def __init__(self, compact_display: bool = False):
        super().__init__({"stroke": "darkred", "stroke_width": PhysSkin._STROKE_WIDTH},
                         {"width": 2,
                          "fill": "lightpink",
                          "stroke_style": {"stroke": "darkred", "stroke_width": 1}},
                         compact_display)
        self.style[ModeType.CLASSICAL] = {"stroke": "orange", "stroke_width": PhysSkin._STROKE_WIDTH}

    @dispatch(AComponent)
    def get_width(self, c) -> int:
        """Absolute fallback"""
        return 1

    @dispatch(cp.Unitary)
    def get_width(self, c) -> int:
        return c.m

    @dispatch((Circuit, cp.BS, cp.PBS))
    def get_width(self, c) -> int:
        return 2

    @dispatch(AFFConfigurator)
    def get_width(self, c) -> int:
        return self.get_width(c.circuit_template())

    @dispatch((cp.PS, nu.TD, cp.PERM, cp.WP, cp.PR, nu.LC))
    def get_width(self, c) -> int:
        return 1

    @dispatch(cp.Barrier)
    def get_width(self, c) -> int:
        return 1 if c.visible else 0

    @dispatch(AComponent)
    def get_shape(self, c):
        return self.default_shape

    @dispatch(cp.BS)
    def get_shape(self, c):
        return self.bs_shape

    @dispatch(cp.PS)
    def get_shape(self, c):
        return self.ps_shape

    @dispatch(cp.PBS)
    def get_shape(self, c):
        return self.pbs_shape

    @dispatch(nu.TD)
    def get_shape(self, c):
        return self.td_shape

    @dispatch(cp.Unitary)
    def get_shape(self, c):
        return self.unitary_shape

    @dispatch(cp.PERM)
    def get_shape(self, c):
        return self.perm_shape

    @dispatch((cp.WP, cp.HWP, cp.QWP))
    def get_shape(self, c):
        return self.wp_shape

    @dispatch(cp.PR)
    def get_shape(self, c):
        return self.pr_shape

    @dispatch(cp.Barrier)
    def get_shape(self, c):
        return self.barrier_shape

    @dispatch(nu.LC)
    def get_shape(self, c):
        return self.lc_shape

    @dispatch(AFFConfigurator)
    def get_shape(self, c):
        return self.ffconf_shape

    @dispatch(Port, PortLocation)
    def get_shape(self, port, location):
        if location == PortLocation.INPUT:
            return self.port_shape_in
        return self.port_shape_out

    @dispatch(Herald, PortLocation)
    def get_shape(self, herald, location):
        if location == PortLocation.INPUT:
            return self.herald_shape_in
        return self.herald_shape_out

    @dispatch(IDetector)
    def get_shape(self, detector):
        if detector.type == DetectionType.PNR:
            return self.pnr_detector_shape
        elif detector.type == DetectionType.Threshold:
            return self.threshold_detector_shape
        elif detector.type == DetectionType.PPNR:
            return self.ppnr_detector_shape
        raise TypeError(f"Unknown detector type: {detector.type}")

    def ffconf_shape(self, comp: AFFConfigurator, canvas, mode_style):
        w = self.get_width(comp)
        for i in range(comp.m):
            canvas.add_mpath(["M", 0, 25 + i * 50, "l", 50 * w, 0], **self.style[ModeType.CLASSICAL])

        # Control wire between the feed-forward configurator and the configured circuit
        offset_sign = math.copysign(1, comp.circuit_offset)
        origin = [w * 25, 25 + offset_sign*15]
        destination = [w * 25, 40 + offset_sign*15 + 50 * comp.circuit_offset]
        if offset_sign > 0:  # Move to the bottom of the ff configurator block if offset is "to the bottom"
            origin[1] += (comp.m - 1)*50
            destination[1] += (comp.m - 1)*50
        canvas.add_mline(origin + destination, stroke="white", stroke_width=PhysSkin._STROKE_WIDTH+2)
        canvas.add_mline(origin + destination, **self.style[ModeType.CLASSICAL], stroke_dasharray="4,4")
        origin[1] += offset_sign * 8
        arrow_size = 3
        for side in [-1, 1]:
            canvas.add_mline(origin + [origin[0] + side * arrow_size, origin[1] - offset_sign*arrow_size],
                             **self.style[ModeType.CLASSICAL])

        # The actual component
        canvas.add_rect((5, 10), 50 * w - 10, 50 * comp.m - 20, fill="lightgreen")
        canvas.add_text((w * 25, 30 + 50*(comp.m-1)/2), size=10, ta="middle", text=comp.name)

    def port_shape_in(self, port, canvas, mode_style):
        canvas.add_rect((-2, 15), 12, 50*port.m - 30, fill="lightgray")
        if port.name:
            canvas.add_text((-2, 50*port.m - 9), text='[' + port.name + ']', size=6, ta="left", fontstyle="italic")

    def port_shape_out(self, port, canvas, mode_style):
        canvas.add_rect((15, 15), 12, 50*port.m - 30, fill="lightgray")
        if port.name:
            canvas.add_text((27, 50*port.m - 9), text='[' + port.name + ']', size=6, ta="right", fontstyle="italic")

    def pnr_detector_shape(self, detector, canvas, mode_style):
        canvas.add_mpath(["M", 0, 25, "l", 25, 0], **self.style[ModeType.PHOTONIC])
        if mode_style[0] is not None:
            canvas.add_mpath(["M", 25, 25, "l", 25, 0], **self.style[ModeType.CLASSICAL])
        canvas.add_mpath(ShapeFactory.half_circle_port_out(10, 20), stroke="black", stroke_width=1, fill="lightgray")
        if detector.name:
            canvas.add_text((12, 12), text=detector.name, size=5, ta="left", fontstyle="italic")

    def threshold_detector_shape(self, detector, canvas, mode_style):
        canvas.add_mpath(["M", 0, 25, "l", 25, 0], **self.style[ModeType.PHOTONIC])
        if mode_style[0] is not None:
            canvas.add_mpath(["M", 25, 25, "l", 25, 0], **self.style[ModeType.CLASSICAL])
        canvas.add_mpath(ShapeFactory.triangle_port_out(10, 14), stroke="black", stroke_width=1, fill="lightgray")
        if detector.name:
            canvas.add_text((12, 12), text=detector.name, size=5, ta="left", fontstyle="italic")

    def ppnr_detector_shape(self, detector, canvas, mode_style):
        canvas.add_mpath(["M", 0, 25, "l", 25, 0], **self.style[ModeType.PHOTONIC])
        if mode_style[0] is not None:
            canvas.add_mpath(["M", 25, 25, "l", 25, 0], **self.style[ModeType.CLASSICAL])
        canvas.add_mpath(ShapeFactory.polygon_port_out(10, 12), stroke="black", stroke_width=1, fill="lightgray")
        if detector.name:
            canvas.add_text((12, 12), text=detector.name, size=5, ta="left", fontstyle="italic")

    def default_shape(self, circuit, canvas, mode_style):
        """
        Default shape is a gray box
        """
        w = self.get_width(circuit)
        content = self._get_display_content(circuit)
        for i in range(circuit.m):
            canvas.add_mpath(["M", 0, 25 + i*50, "l", 50*w, 0], **self.style[mode_style[i]])
        canvas.add_rect((5, 5), 50*w - 10, 50*circuit.m - 10, fill="gray")
        canvas.add_text((25*w, 25*circuit.m), size=7, ta="middle", text=content)

    @staticmethod
    def _reflective_side(theta, convention: cp.BSConvention) -> int:
        """
        Return the reflective side of a beam splitter given a theta parameter and a BSConvention
        1 means top
        -1 means bottom
        0 means undecided
        """
        if convention == cp.BSConvention.Rx:
            return 1  # top
        if not theta.defined:
            return 0
        theta = float(theta)
        if convention == cp.BSConvention.Ry:
            return 1 if theta < 2*math.pi else -1
        elif convention == cp.BSConvention.H:
            return -1 if round(theta/2/math.pi) % 2 else 1

    def bs_shape(self, bs, canvas, mode_style):
        split_content = self._get_display_content(bs).split("\n")
        head_content = "\n".join([s for s in split_content
                                  if s.startswith("R=") or s.startswith("theta=")])
        bottom_content_list = [s for s in split_content
                               if not s.startswith("R=") and not s.startswith("theta=")]
        bottom_nline = len(bottom_content_list)
        bottom_size = 7 if bottom_nline < 3 else 6
        mode_style = self.style[ModeType.PHOTONIC]
        canvas.add_mline([0, 25, 28, 25, 47, 44], stroke_linejoin="round", **mode_style)
        canvas.add_mline([53, 44, 72, 25, 100, 25], stroke_linejoin="round", **mode_style)
        canvas.add_mline([0, 75, 28, 75, 47, 56], stroke_linejoin="round", **mode_style)
        canvas.add_mline([53, 56, 72, 75, 100, 75], stroke_linejoin="round", **mode_style)
        canvas.add_rect((25, 43), 50, 14, fill="black")
        canvas.add_text((50, 80+5*bottom_nline), '\n'.join(bottom_content_list).replace('phi_', 'Φ_'),
                        size=bottom_size, ta="middle")
        canvas.add_text((50, 26), head_content.replace('theta=', 'Θ='), size=7, ta="middle")
        # Choose the side of the gray rectangle in beam splitter representation

        r_side = self._reflective_side(bs.param('theta'), bs.convention)
        if r_side == 1:
            canvas.add_rect((25, 43), 50, 4, fill="lightgray")
        elif r_side == -1:
            canvas.add_rect((25, 53), 50, 4, fill="lightgray")
        # Add BS convention badge
        canvas.add_rect((68, 50), 10, 10, fill=bs_convention_color(bs.convention))
        canvas.add_text((73, 57), bs.convention.name, size=6, ta="middle")

    def ps_shape(self, circuit, canvas, mode_style):
        canvas.add_mline([0, 25, 50, 25], **self.style[ModeType.PHOTONIC])
        canvas.add_polygon([5, 40, 14, 40, 28, 10, 19, 10, 5, 40, 14, 40],
                           stroke="black", fill="gray", stroke_width=1, stroke_linejoin="miter")
        content = self._get_display_content(circuit).replace("phi=", "Φ=")
        canvas.add_text((22, 38), text=content, size=7, ta="left")

    def lc_shape(self, circuit, canvas, mode_style):
        style = {'stroke': 'black', 'stroke_width': 1}
        canvas.add_mline([0, 25, 50, 25], **self.style[ModeType.PHOTONIC])
        canvas.add_mline([25, 25, 25, 32], **style)
        canvas.add_mline([15, 32, 35, 32], **style)
        canvas.add_mline([18, 34, 32, 34], **style)
        canvas.add_mline([21, 36, 29, 36], **style)
        canvas.add_mline([24, 38, 26, 38], **style)
        canvas.add_rect((22, 22), 6, 6, fill="gray")
        canvas.add_text((6, 20), text=self._get_display_content(circuit), size=7, ta="left")

    def pbs_shape(self, circuit, canvas, mode_style):
        style = self.style[ModeType.PHOTONIC]
        canvas.add_mline([0, 25, 28, 25, 37.5, 37.5], **style, stroke_linejoin="round")
        canvas.add_mline([62.5, 37.5, 72, 25, 100, 25], **style, stroke_linejoin="round")
        canvas.add_mline([0, 75, 28, 75, 37.5, 62.5], **style, stroke_linejoin="round")
        canvas.add_mline([62.5, 62.5, 72, 75, 100, 75], **style, stroke_linejoin="round")
        canvas.add_mline([62.5, 62.5, 72, 75, 100, 75], **style, stroke_linejoin="round")
        canvas.add_polygon([25, 50, 50, 24, 75, 50, 50, 76, 25, 50], stroke="black", stroke_width=1, fill="gray")
        canvas.add_mline([25, 50, 75, 50], stroke="black", stroke_width=1)
        canvas.add_text((50, 86), text=self._get_display_content(circuit), size=7, ta="middle")

    def td_shape(self, circuit, canvas, mode_style):
        style = self.style[ModeType.PHOTONIC]
        for h_shift in [0, 9, 18]:
            canvas.add_circle((34 - h_shift, 14), 11, stroke_width=5, fill=None, stroke="white")
            canvas.add_circle((34 - h_shift, 14), 11, fill=None, **style)
        canvas.add_mline([0, 25, 19, 25], stroke="white", stroke_width=5)
        canvas.add_mline([0, 25, 19, 25], **style)
        canvas.add_mline([34, 25, 50, 25], stroke="white", stroke_width=5)
        canvas.add_mline([32, 25, 50, 25], **style)
        canvas.add_text((25, 38), self._get_display_content(circuit), 7, "middle")

    def unitary_shape(self, circuit, canvas, mode_style):
        m = circuit.m
        for i in range(m):
            canvas.add_mpath(["M", 0, 25 + i*50, "l", 50*m, 0], **self.style[ModeType.PHOTONIC])
        canvas.add_rect((5, 5), 50*m-10, 50*m-10, fill="lightyellow")
        canvas.add_text((25*m, 25*m), size=10, ta="middle", text=circuit.name)

    def barrier_shape(self, barrier, canvas, mode_style):
        if not barrier.visible:
            return

        m = barrier.m
        if canvas.background_color is None:
            canvas.add_rect((10, 10), 30, 50 * m - 20, fill="whitesmoke", stroke="whitesmoke")
        for i in range(m):
            style = self.style[mode_style[i]]
            if style["stroke"]:
                canvas.add_mpath(["M", 0, 25 + i*50, "l", 50, 0], **style)
        canvas.add_rect((24, 10), 2, 50 * m - 20, fill="dimgrey", stroke="dimgrey")

    def perm_shape(self, circuit, canvas, mode_style):
        for an_input, an_output in enumerate(circuit.perm_vector):
            style = self.style[mode_style[an_input]]
            if style['stroke']:
                canvas.add_mline([3, 25+an_input*50, 47, 25+an_output*50],
                                 stroke="white", stroke_width=style["stroke_width"]+3)
                canvas.add_mline([0, 25+an_input*50, 3, 25+an_input*50, 47, 25+an_output*50, 50, 25+an_output*50],
                                 **style)

    def wp_shape(self, circuit, canvas, mode_style):
        params = self._get_display_content(circuit).replace("xsi=", "ξ=").replace("delta=", "δ=").split("\n")
        canvas.add_mline([0, 25, 50, 25], **self.style[ModeType.PHOTONIC])
        canvas.add_rect((13, 7), width=14, height=36, fill="gray",
                        stroke_width=1, stroke="black", stroke_linejoin="miter")
        canvas.add_mline([20, 7, 20, 43], stroke="black", stroke_width=1)
        canvas.add_text((28.5, 36), text=params[0], size=7, ta="left")
        canvas.add_text((28.5, 45), text=params[1], size=7, ta="left")

    def pr_shape(self, circuit, canvas, mode_style):
        canvas.add_mline([0, 25, 15, 25], **self.style[ModeType.PHOTONIC])
        canvas.add_mline([35, 25, 50, 25], **self.style[ModeType.PHOTONIC])
        canvas.add_rect((14, 14), width=22, height=22, stroke="black", fill="lightgray",
                        stroke_width=1, stroke_linejoin="miter")
        canvas.add_mpath(ShapeFactory.pr_mpath, fill="black")
        canvas.add_text((25, 45), text=self._get_display_content(circuit).replace("delta=", "δ="), size=7, ta="middle")

    def subcircuit_shape(self, circuit, canvas, mode_style):
        w = self.style_subcircuit['width']
        for idx in range(circuit.m):
            canvas.add_mline([0, 50*idx+25, w*50, 50*idx+25], **self.style[ModeType.PHOTONIC])
        canvas.add_rect((2.5, 2.5), w*50 - 5, 50*circuit.m - 5,
                        fill=self.style_subcircuit['fill'], **self.style_subcircuit['stroke_style'])
        title = circuit.name.upper().split(" ")
        canvas.add_text((10, 8+8*len(title)), "\n".join(title), 8, fontstyle="bold")

    def herald_shape_in(self, herald, canvas, mode_style):
        canvas.add_mpath(ShapeFactory.half_circle_port_in(10), stroke="black", stroke_width=1, fill="white")
        if herald.name:
            canvas.add_text((13, 41), text='[' + herald.name + ']', size=6, ta="middle", fontstyle="italic")
        canvas.add_text((17, 28), text=str(herald.expected), size=7, ta="middle")

    def herald_shape_out(self, herald, canvas, mode_style):
        if herald.detector_type is None or herald.detector_type == DetectionType.PNR:
            shape = ShapeFactory.half_circle_port_out(10)
        elif herald.detector_type == DetectionType.PPNR:
            shape = ShapeFactory.polygon_port_out(10)
        elif herald.detector_type == DetectionType.Threshold:
            shape = ShapeFactory.triangle_port_out(10)
        else:
            raise TypeError(f"Unknown detector type: {herald.detector_type}")
        canvas.add_mpath(shape, stroke="black", stroke_width=1, fill="white")
        if herald.name:
            canvas.add_text((13, 11), text='[' + herald.name + ']', size=6, ta="middle", fontstyle="italic")
        canvas.add_text((8, 28), text=str(herald.expected), size=7, ta="middle")
