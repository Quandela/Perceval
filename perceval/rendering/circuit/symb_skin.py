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

from perceval.components import AComponent, AFFConfigurator, Circuit, Port, PortLocation, Herald, IDetector,\
    unitary_components as cp,\
    non_unitary_components as nu
from ._canvas_shapes import ShapeFactory
from .abstract_skin import ASkin, ModeType
from .skin_common import bs_convention_color


class SymbSkin(ASkin):
    def __init__(self, compact_display: bool = False):
        super().__init__({"stroke": "black", "stroke_width": 1},
                         {"width": 1,
                          "fill": "white",
                          "stroke_style": {"stroke": "black", "stroke_width": 1}},
                         compact_display)
        self.style[ModeType.CLASSICAL] = {"stroke": "gray", "stroke_width": 3}

    @dispatch(AComponent)
    def get_width(self, c) -> int:
        """Absolute fallback"""
        return 1

    @dispatch(AFFConfigurator)
    def get_width(self, c) -> int:
        return self.get_width(c.circuit_template())

    @dispatch(cp.Unitary)
    def get_width(self, c) -> int:
        return c.m

    @dispatch(Circuit)
    def get_width(self, c) -> int:
        return 2

    @dispatch(cp.Barrier)
    def get_width(self, c: cp.Barrier) -> int:
        return 1 if c.visible else 0

    @dispatch((cp.BS, cp.PBS))
    def get_width(self, c) -> int:
        w = 1 if self._compact else 2
        return w

    @dispatch((cp.PS, nu.TD, cp.PERM, cp.WP, cp.PR, nu.LC))
    def get_width(self, c) -> int:
        return 1

    @dispatch(AComponent)
    def get_shape(self, c):
        return self.default_shape

    @dispatch(AFFConfigurator)
    def get_shape(self, c):
        return self.ffconf_shape

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

    @dispatch(cp.WP)
    def get_shape(self, c):
        return self.wp_shape

    @dispatch(cp.HWP)
    def get_shape(self, c):
        return self.hwp_shape

    @dispatch(cp.QWP)
    def get_shape(self, c):
        return self.qwp_shape

    @dispatch(cp.PR)
    def get_shape(self, c):
        return self.pr_shape

    @dispatch(cp.Barrier)
    def get_shape(self, c):
        return self.barrier_shape

    @dispatch(nu.LC)
    def get_shape(self, c):
        return self.lc_shape

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

    def ffconf_shape(self, comp: AFFConfigurator, canvas, mode_style):
        w = self.get_width(comp)
        for i in range(comp.m):
            canvas.add_mpath(["M", 0, 25 + i * 50, "l", 50 * w, 0], **self.style[ModeType.CLASSICAL])

        # Control wire between the feed-forward configurator and the configured circuit
        offset_sign = math.copysign(1, comp.circuit_offset)
        origin = [w * 25, 25 + offset_sign * 15]
        destination = [w * 25, 40 + offset_sign * 15 + 50 * comp.circuit_offset]
        if offset_sign > 0:  # Move to the bottom of the ff configurator block if offset is "to the bottom"
            origin[1] += (comp.m - 1)*50
            destination[1] += (comp.m - 1)*50
        canvas.add_mline(origin + destination, stroke="white", stroke_width=4.5)
        canvas.add_mline(origin + destination, **self.style[ModeType.CLASSICAL], stroke_dasharray="9,5")
        origin[1] += offset_sign * 8
        arrow_size = 5
        for side in [-1, 1]:
            canvas.add_mline(origin + [origin[0] + side * arrow_size, origin[1] - offset_sign * arrow_size],
                             **self.style[ModeType.CLASSICAL])

        # The actual component
        canvas.add_rect((5, 10), 50 * w - 10, 50 * comp.m - 20, fill="honeydew")
        canvas.add_text((w * 25, 30 + 50*(comp.m-1)/2), size=10, ta="middle", text=comp.name)

    @dispatch(IDetector)
    def get_shape(self, detector):
        return self.detector_shape

    def default_shape(self, circuit, canvas, mode_style):
        """
        Default shape is a gray box
        """
        w = self.get_width(circuit)
        content = self._get_display_content(circuit)
        for i in range(circuit.m):
            canvas.add_mpath(["M", 0, 25 + i*50, "l", 50*w, 0], **self.style[mode_style[i]])
        canvas.add_rect((5, 5), 50*w - 10, 50*circuit.m - 10, fill="lightgray")
        canvas.add_text((25*w, 25*circuit.m), size=7, ta="middle", text=content)

    def bs_shape(self, bs, canvas, mode_style):
        canvas.add_mpath(ShapeFactory.bs_symbolic_mpath(self._compact), **self.style[ModeType.PHOTONIC])
        content = self._get_display_content(bs).replace('phi', 'Φ').replace('theta=', 'Θ=')
        canvas.add_text((25 if self._compact else 50, 38), content, 7, "middle")
        # Add BS convention badge
        canvas.add_rect((35 if self._compact else 72, 53), 10, 10, fill=bs_convention_color(bs.convention))
        canvas.add_text((40 if self._compact else 77, 60), bs.convention.name, size=6, ta="middle")

    def ps_shape(self, circuit, canvas, mode_style):
        canvas.add_mpath(["M", 0, 25, "h", 20, "m", 10, 0, "h", 20], **self.style[ModeType.PHOTONIC])
        canvas.add_rect((15, 15), 20, 20, stroke="black", stroke_width=1, fill="lightgray")
        content = self._get_display_content(circuit).replace("phi=", "Φ=")
        canvas.add_text((25, 44), text=content, size=7, ta="middle")

    def lc_shape(self, circuit, canvas, mode_style):
        style = {'stroke': 'black', 'stroke_width': 1}
        canvas.add_mline([0, 25, 50, 25], **self.style[ModeType.PHOTONIC])
        canvas.add_mline([25, 25, 25, 32], **style)
        canvas.add_mline([15, 32, 35, 32], **style)
        canvas.add_mline([18, 34, 32, 34], **style)
        canvas.add_mline([21, 36, 29, 36], **style)
        canvas.add_mline([24, 38, 26, 38], **style)
        canvas.add_rect((22, 22), 6, 6, fill="white")
        canvas.add_text((6, 20), text=self._get_display_content(circuit), size=7, ta="left")

    def pbs_shape(self, circuit, canvas, mode_style):
        if self._compact:
            path_data1 = ["M", 0, 25.1, "h", 11.049, "m", -11.049, 50, "h", 10.9375, "m", 27.9029, -50, "h",
                          11.1596,
                          "m", -11.3283, 50, "h", 11.3283, "m", -11.3283, 0, "c", -10.0446, 0, -17.5781, -50,
                          -27.7341,
                          -50, "m", 27.9029, 0, "c", -10.7156, 0, -17.7467, 50, -27.7914, 50]
            path_data2 = ["M", 30, 50, "l", -4.7404, -5.2543, "l", -4.7404, 5.2543, "l", 4.7404, 5.2543, "l",
                          4.7404, -5.2543, "z", "m", 0.175, 0, "h", -9.6, "z"]
        else:
            path_data1 = ["M", 0, 25.1, "h", 22.0981, "m", -22.0981, 50, "h", 21.8751, "m", 55.8057, -50, "h",
                          22.3192,
                          "m", -22.6566, 50, "h", 22.6566, "m", -22.6566, 0, "c", -20.0892, 0, -35.1561, -50,
                          -55.4683,
                          -50, "m", 55.8057, 0, "c", -21.4311, 0, -35.4935, 50, -55.5827, 50]
            path_data2 = ["M", 59, 50, "l", -9.4807, -10.5087, "l", -9.4807, 10.5087, "l", 9.4807, 10.5087, "l",
                          9.4807,
                          -10.5087, "z", "m", 0.35, 0, "h", -19.2, "z"]
        canvas.add_mpath(path_data1, **self.style[ModeType.PHOTONIC])
        canvas.add_mpath(path_data2, stroke_width=1, fill="#fff")
        canvas.add_text((25 if self._compact else 50, 86), text=self._get_display_content(circuit), size=7, ta="middle")

    def td_shape(self, circuit, canvas, mode_style):
        stroke = self.style[ModeType.PHOTONIC]['stroke']
        for h_shift in [0, 9, 18]:
            canvas.add_circle((34 - h_shift, 14), 11, stroke="white", stroke_width=3)
            canvas.add_circle((34 - h_shift, 14), 11, stroke=stroke, stroke_width=2)
        canvas.add_mline([0, 25, 17, 25], stroke="white", stroke_width=3)
        canvas.add_mline([0, 25, 19, 25], stroke=stroke, stroke_width=2)
        canvas.add_mline([34, 25, 50, 25], stroke="white", stroke_width=3)
        canvas.add_mline([32, 25, 50, 25], stroke=stroke, stroke_width=2)
        canvas.add_text((25, 38), text=self._get_display_content(circuit), size=7, ta="middle")

    def unitary_shape(self, circuit, canvas, mode_style):
        w = circuit.m
        for i in range(w):
            canvas.add_mpath(["M", 0, 25 + i*50, "l", 50*w, 0], **self.style[ModeType.PHOTONIC])
        shape = ShapeFactory.rounded_corner_square(6.25*w, 6)
        canvas.add_mpath(shape, **self.style[ModeType.PHOTONIC], fill="lightyellow")
        canvas.add_text((25*w, 25*w), size=10, ta="middle", text=circuit.name)

    def barrier_shape(self, barrier: cp.Barrier, canvas, mode_style):
        if not barrier.visible:
            return

        m = barrier.m
        canvas.add_mline((24, 10, 24, 50 * m - 16), stroke_width=1, stroke="lightgray")
        for i in range(m):
            style = self.style[mode_style[i]]
            if style["stroke"]:
                canvas.add_mpath(["M", 0, 25 + i*50, "l", 50, 0], **style)

    def perm_shape(self, circuit, canvas, mode_style):
        for an_input, an_output in enumerate(circuit.perm_vector):
            style = self.style[mode_style[an_input]]
            if style['stroke']:
                canvas.add_mpath(["M", 0, 24.8 + an_input * 50,
                                  "C", 20, 25 + an_input * 50, 30, 25 + an_output * 50, 50, 25 + an_output * 50],
                                 stroke="white", stroke_width=2)
                canvas.add_mpath(["M", 0, 25 + an_input * 50,
                                  "C", 20, 25 + an_input * 50, 30, 25 + an_output * 50, 50, 25 + an_output * 50],
                                 **style)

    def wp_shape(self, circuit, canvas, mode_style):
        params = self._get_display_content(circuit).replace("xsi=", "ξ=").replace("delta=", "δ=").split("\n")
        style = self.style[ModeType.PHOTONIC]
        canvas.add_mpath(["M", 0, 25, "h", 15, "m", 21, 0, "h", 15], **style)
        canvas.add_mpath(["M", 15, 45, "h", 21, "v", -40, "h", -21, "z"], **style)
        canvas.add_text((25, 55), text=params[0], size=7, ta="middle")
        canvas.add_text((25, 65), text=params[1], size=7, ta="middle")

    def hwp_shape(self, circuit, canvas, mode_style):
        params = self._get_display_content(circuit).replace("xsi=", "ξ=").replace("delta=", "δ=").split("\n")
        canvas.add_mpath(["M", 0, 25, "v", 0, "h", 0, "h", 50], **self.style[ModeType.PHOTONIC])
        canvas.add_mpath(["M", 20, 0, "v", 50], stroke="black", stroke_width=2)
        canvas.add_mpath(["M", 30, 0, "v", 50], stroke="black", stroke_width=2)
        canvas.add_text((25, 60), text=params[0], size=7, ta="middle")

    def qwp_shape(self, circuit, canvas, mode_style):
        params = self._get_display_content(circuit).replace("xsi=", "ξ=").replace("delta=", "δ=").split("\n")
        canvas.add_mpath(["M", 0, 25, "v", 0, "h", 0, "h", 50], **self.style[ModeType.PHOTONIC])
        canvas.add_mpath(["M", 25, 0, "v", 50], stroke="black", stroke_width=2)
        canvas.add_text((25, 60), text=params[0], size=7, ta="middle")

    def pr_shape(self, circuit, canvas, mode_style):
        canvas.add_mpath(["M", 0, 25, "h", 15, "m", 22, 0, "h", 15], **self.style[ModeType.PHOTONIC])
        canvas.add_mpath(["M", 15, 36, "h", 22, "v", -22, "h", -22, "z"], stroke="black", stroke_width=1)
        canvas.add_mpath(ShapeFactory.pr_mpath, fill="black", stroke_width=0.1)
        canvas.add_text((27, 50), text=self._get_display_content(circuit).replace("delta=", "δ="), size=7, ta="middle")

    def subcircuit_shape(self, circuit, canvas, mode_style):
        w = self.style_subcircuit['width']
        for idx in range(circuit.m):
            canvas.add_mline([0, 50*idx+25, w*50, 50*idx+25], **self.style[ModeType.PHOTONIC])
        canvas.add_rect((2.5, 2.5), w*50 - 5, 50*circuit.m - 5,
                        fill=self.style_subcircuit['fill'], **self.style_subcircuit['stroke_style'])
        title = circuit.name.upper().split(" ")
        canvas.add_text((10, 8 + 8 * len(title)), "\n".join(title), 8, fontstyle="bold")

    def herald_shape_in(self, herald, canvas, mode_style):
        canvas.add_mpath(ShapeFactory.half_circle_port_in(10), stroke="black", stroke_width=1, fill="white")
        if herald.name:
            canvas.add_text((13, 41), text='[' + herald.name + ']', size=6, ta="middle", fontstyle="italic")
        canvas.add_text((17, 28), text=str(herald.expected), size=7, ta="middle")

    def herald_shape_out(self, herald, canvas, mode_style):
        canvas.add_mpath(ShapeFactory.half_circle_port_out(10), stroke="black", stroke_width=1, fill="white")
        if herald.name:
            canvas.add_text((13, 11), text='[' + herald.name + ']', size=6, ta="middle", fontstyle="italic")
        canvas.add_text((8, 28), text=str(herald.expected), size=7, ta="middle")

    def port_shape_in(self, port, canvas, mode_style):
        canvas.add_rect((-2, 15), 12, 50*port.m - 30, fill="white")
        if port.name:
            canvas.add_text((-2, 50*port.m - 9), text='[' + port.name + ']', size=6, ta="left", fontstyle="italic")

    def port_shape_out(self, port, canvas, mode_style):
        canvas.add_rect((15, 15), 12, 50*port.m - 30, fill="white")
        if port.name:
            canvas.add_text((27, 50*port.m - 9), text='[' + port.name + ']', size=6, ta="right", fontstyle="italic")

    def detector_shape(self, detector, canvas, mode_style):
        canvas.add_mpath(["M", 0, 25, "l", 25, 0], **self.style[ModeType.PHOTONIC])
        if mode_style[0] is not None:
            canvas.add_mpath(["M", 25, 25, "l", 25, 0], **self.style[ModeType.CLASSICAL])
        canvas.add_mpath(ShapeFactory.half_circle_port_out(10, 20), stroke="black", stroke_width=1, fill="white")
        if detector.name:
            canvas.add_text((12, 12), text=detector.name, size=5, ta="left", fontstyle="italic")
