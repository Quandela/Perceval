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

from multipledispatch import dispatch

from perceval.components import AComponent, Circuit, Port, PortLocation, Herald,\
    unitary_components as cp,\
    non_unitary_components as nu
from .abstract_skin import ASkin, ModeStyle
from .skin_common import bs_convention_color


class SymbSkin(ASkin):
    def __init__(self, compact_display: bool = False):
        super().__init__({"stroke": "black", "stroke_width": 1},
                         {"width": 1,
                          "fill": "white",
                          "stroke_style": {"stroke": "black", "stroke_width": 1}},
                         compact_display)

    @dispatch(AComponent)
    def get_width(self, c) -> int:
        """Absolute fallback"""
        return 1

    @dispatch(cp.Unitary)
    def get_width(self, c) -> int:
        return c.m

    @dispatch(Circuit)
    def get_width(self, c) -> int:
        return 2

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

    def default_shape(self, circuit, canvas, content, mode_style, **opts):
        """
        Default shape is a gray box
        """
        w = self.get_width(circuit)
        for i in range(circuit.m):
            canvas.add_mpath(["M", 0, 25 + i*50, "l", 50*w, 0], **self.style[ModeStyle.PHOTONIC])
        canvas.add_rect((5, 5), 50*w - 10, 50*circuit.m - 10, fill="lightgray")
        canvas.add_text((25*w, 25*circuit.m), size=7, ta="middle", text=content)

    def bs_shape(self, bs, canvas, content, mode_style, **opts):
        if self._compact:
            path_data = ["M", 6.4721, 25.0002, "c", 6.8548, 0, 6.8241, 24.9998, 13.6789, 24.9998, "m", 0.0009, 0, "c",
                         -6.8558, 0, -6.825, 24.9998, -13.6799, 24.9998, "m", 13.6799, -24.9998, "h", 10.9423, "m", 0,
                         0, "c", 6.8558, 0, 6.825, -24.9998, 13.6799, -24.9998, "m", -13.6799, 24.9998, "c", 6.8558, 0,
                         6.825, 24.9998, 13.6799, 24.9998, "m", -44.7741, -49.9998, "h", 6.5, "m", 0.0009, 49.9998, "h",
                         -6.5009, "m", 43.8227, 0, "h", 6.1773, "m", -6.4028, -50, "h", 6.4028]
        else:
            path_data = ["M", 12.9442, 25.0002, "c", 13.7096, 0, 13.6481, 24.9998, 27.3577, 24.9998, "m", 0.0019, 0,
                         "c", -13.7116, 0, -13.65, 24.9998, -27.3597, 24.9998, "m", 27.3597, -24.9998, "h", 21.8846,
                         "m", 0, 0, "c", 13.7116, 0, 13.65, -24.9998, 27.3597, -24.9998, "m", -27.3597, 24.9998, "c",
                         13.7116, 0, 13.65, 24.9998, 27.3597, 24.9998, "m", -89.5481, -49.9998, "h", 13, "m", 0.0019,
                         49.9998, "h", -13.0019, "m", 87.6453, 0, "h", 12.3547, "m", -12.8056, -50, "h", 12.8056]
        canvas.add_mpath(path_data, **self.style[ModeStyle.PHOTONIC])
        canvas.add_text((25 if self._compact else 50, 38),
                        content.replace('phi', 'Φ').replace('theta=', 'Θ='),
                        7, "middle")
        # Add BS convention badge
        canvas.add_rect((35 if self._compact else 72, 53), 10, 10, fill=bs_convention_color(bs.convention))
        canvas.add_text((40 if self._compact else 77, 60), bs.convention.name, size=6, ta="middle")

    def ps_shape(self, circuit, canvas, content, mode_style, **opts):
        canvas.add_mpath(["M", 0, 25, "h", 20, "m", 10, 0, "h", 20], **self.style[ModeStyle.PHOTONIC])
        canvas.add_mpath(["M", 15, 35, "h", 20, "v", -20, "h", -20, "z"],
                         stroke="black", stroke_width=1, fill="lightgray")
        canvas.add_text((25, 44), text=content.replace("phi=", "Φ="), size=7, ta="middle")

    def lc_shape(self, circuit, canvas, content, mode_style, **opts):
        style = {'stroke': 'black', 'stroke_width': 1}
        canvas.add_mline([0, 25, 50, 25], **self.style[ModeStyle.PHOTONIC])
        canvas.add_mline([25, 25, 25, 32], **style)
        canvas.add_mline([15, 32, 35, 32], **style)
        canvas.add_mline([18, 34, 32, 34], **style)
        canvas.add_mline([21, 36, 29, 36], **style)
        canvas.add_mline([24, 38, 26, 38], **style)
        canvas.add_rect((22, 22), 6, 6, fill="white")
        canvas.add_text((6, 20), text=content, size=7, ta="left")

    def pbs_shape(self, circuit, canvas, content, mode_style, **opts):
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
        canvas.add_mpath(path_data1, **self.style[ModeStyle.PHOTONIC])
        canvas.add_mpath(path_data2, stroke_width=1, fill="#fff")
        canvas.add_text((25 if self._compact else 50, 86), text=content, size=7, ta="middle")

    def td_shape(self, circuit, canvas, content, mode_style, **opts):
        stroke = self.style[ModeStyle.PHOTONIC]['stroke']
        canvas.add_circle((34, 14), 11, stroke="white", stroke_width=3)
        canvas.add_circle((34, 14), 11, stroke=stroke, stroke_width=2)
        canvas.add_circle((25, 14), 11, stroke="white", stroke_width=3)
        canvas.add_circle((25, 14), 11, stroke=stroke, stroke_width=2)
        canvas.add_circle((16, 14), 11, stroke="white", stroke_width=3)
        canvas.add_circle((16, 14), 11, stroke=stroke, stroke_width=2)
        canvas.add_mline([0, 25, 17, 25], stroke="white", stroke_width=3)
        canvas.add_mline([0, 25, 19, 25], stroke=stroke, stroke_width=2)
        canvas.add_mline([34, 25, 50, 25], stroke="white", stroke_width=3)
        canvas.add_mline([32, 25, 50, 25], stroke=stroke, stroke_width=2)
        canvas.add_text((25, 38), text=content.replace("t=", ""), size=7, ta="middle")

    def unitary_shape(self, circuit, canvas, content, mode_style, **opts):
        w = circuit.m
        for i in range(circuit.m):
            canvas.add_mpath(["M", 0, 25 + i*50, "l", 50*w, 0], **self.style[ModeStyle.PHOTONIC])
        radius = 6.25 * w  # Radius of the rounded corners
        canvas.add_mpath(
            ["M", 0, radius, "c", 0, 0, 0, -radius, radius, -radius, "l", 6 * radius, 0, "c", radius, 0, radius, radius,
             radius, radius, "l", 0, 6 * radius, "c", 0, 0, 0, radius, -radius, radius, "l", -6 * radius, 0, "c",
             -radius, 0, -radius, -radius, -radius, -radius, "l", 0, -6 * radius],
            **self.style[ModeStyle.PHOTONIC], fill="lightyellow")
        canvas.add_text((25*w, 25*w), size=10, ta="middle", text=circuit.name)

    def perm_shape(self, circuit, canvas, content, mode_style, **opts):
        for an_input, an_output in enumerate(circuit.perm_vector):
            style = self.style[mode_style[an_input]]
            if style['stroke']:
                canvas.add_mpath(["M", 0, 24.8 + an_input * 50,
                                  "C", 20, 25 + an_input * 50, 30, 25 + an_output * 50, 50, 25 + an_output * 50],
                                 stroke="white", stroke_width=2)
                canvas.add_mpath(["M", 0, 25 + an_input * 50,
                                  "C", 20, 25 + an_input * 50, 30, 25 + an_output * 50, 50, 25 + an_output * 50],
                                 **style)

    def wp_shape(self, circuit, canvas, content, mode_style, **opts):
        params = content.replace("xsi=", "ξ=").replace("delta=", "δ=").split("\n")
        style = self.style[ModeStyle.PHOTONIC]
        canvas.add_mpath(["M", 0, 25, "h", 15, "m", 21, 0, "h", 15], **style)
        canvas.add_mpath(["M", 15, 45, "h", 21, "v", -40, "h", -21, "z"], **style)
        canvas.add_text((25, 55), text=params[0], size=7, ta="middle")
        canvas.add_text((25, 65), text=params[1], size=7, ta="middle")

    def hwp_shape(self, circuit, canvas, content, mode_style, **opts):
        params = content.replace("xsi=", "ξ=").replace("delta=", "δ=").split("\n")
        canvas.add_mpath(["M", 0, 25, "v", 0, "h", 0, "h", 50], **self.style[ModeStyle.PHOTONIC])
        canvas.add_mpath(["M", 20, 0, "v", 50], stroke="black", stroke_width=2)
        canvas.add_mpath(["M", 30, 0, "v", 50], stroke="black", stroke_width=2)
        canvas.add_text((25, 60), text=params[0], size=7, ta="middle")

    def qwp_shape(self, circuit, canvas, content, mode_style, **opts):
        params = content.replace("xsi=", "ξ=").replace("delta=", "δ=").split("\n")
        canvas.add_mpath(["M", 0, 25, "v", 0, "h", 0, "h", 50], **self.style[ModeStyle.PHOTONIC])
        canvas.add_mpath(["M", 25, 0, "v", 50], stroke="black", stroke_width=2)
        canvas.add_text((25, 60), text=params[0], size=7, ta="middle")

    def pr_shape(self, circuit, canvas, content, mode_style, **opts):
        canvas.add_mpath(["M", 0, 25, "h", 15, "m", 22, 0, "h", 15], **self.style[ModeStyle.PHOTONIC])
        canvas.add_mpath(["M", 15, 36, "h", 22, "v", -22, "h", -22, "z"], stroke="black", stroke_width=1)
        canvas.add_mpath(["M", 19, 27, "c", 0.107, 0.131, 0.280, 0.131, 0.387, 0,
                          "l", 2.305, -2.821, "c", 0.107, -0.131, 0.057, -0.237, -0.112, -0.237,
                          "h", -1.22, "c", -0.169, 0, -0.284, -0.135, -0.247, -0.300,
                          "c", 0.629, -2.866, 3.187, -5.018, 6.240, -5.018,
                          "c", 3.524, 0, 6.39, 2.867, 6.390, 6.3902,
                          "c", 0, 3.523, -2.866, 6.39, -6.390, 6.390,
                          "c", -0.422, 0, -0.765, 0.342, -0.765, 0.765,
                          "s", 0.342, 0.765, 0.765, 0.765,
                          "c", 4.367, 0, 7.92, -3.552, 7.920, -7.920,
                          "c", 0, -4.367, -3.552, -7.920, -7.920, -7.920,
                          "c", -3.898, 0, -7.146, 2.832, -7.799, 6.546,
                          "c", -0.029, 0.166, -0.184, 0.302, -0.353, 0.302,
                          "H", 17, "c", -0.169, 0, -0.219, 0.106, -0.112, 0.237,
                          "z"
                          ], fill="black", stroke_width=0.1)
        canvas.add_text((27, 50), text=content.replace("delta=", "δ="), size=7, ta="middle")

    def subcircuit_shape(self, circuit, canvas, content, mode_style, **opts):
        w = self.style_subcircuit['width']
        for idx in range(circuit.m):
            canvas.add_mline([0, 50*idx+25, w*50, 50*idx+25], **self.style[ModeStyle.PHOTONIC])
        canvas.add_rect((2.5, 2.5), w*50 - 5, 50*circuit.m - 5,
                        fill=self.style_subcircuit['fill'], **self.style_subcircuit['stroke_style'])
        title = circuit.name.upper().split(" ")
        canvas.add_text((10, 8 + 8 * len(title)), "\n".join(title), 8, fontstyle="bold")

    def herald_shape_in(self, herald, canvas, content, mode_style, **opts):
        r = 10
        canvas.add_mpath(["M", 7, 25, "c", 0, 0, 0, -r, r, -r,
                          "h", 8, "v", 2 * r, "h", -8,
                          "c", -r, 0, -r, -r, -r, -r, "z"],
                         stroke="black", stroke_width=1, fill="white")
        if herald.name:
            canvas.add_text((13, 41), text='[' + herald.name + ']', size=6, ta="middle", fontstyle="italic")
        canvas.add_text((17, 28), text=str(herald.expected), size=7, ta="middle")

    def herald_shape_out(self, herald, canvas, content, mode_style, **opts):
        r = 10  # Radius of the half-circle
        canvas.add_mpath(["M", 8, 35, "h", -8, "v", -2 * r, "h", 8,
                          "c", 0, 0, r, 0, r, r,
                          "c", 0, r, -r, r, -r, r, "z"],
                         stroke="black", stroke_width=1, fill="white")
        if herald.name:
            canvas.add_text((13, 41), text='[' + herald.name + ']', size=6, ta="middle", fontstyle="italic")
        canvas.add_text((8, 28), text=str(herald.expected), size=7, ta="middle")

    def port_shape_in(self, port, canvas, content, mode_style, **opts):
        m_index = None
        if 'starting_mode' in opts:
            m_index = opts['starting_mode']
        canvas.add_rect((-2, 15), 12, 50*port.m - 30, fill="white")
        if m_index is not None:
            for i in range(port.m):
                canvas.add_text((4, 50 * i + 27), text=str(m_index+i), size=7, ta="middle")
        if port.name:
            canvas.add_text((-2, 50*port.m - 9), text='[' + port.name + ']', size=6, ta="left", fontstyle="italic")

    def port_shape_out(self, port, canvas, content, mode_style, **opts):
        m_index = None
        if 'starting_mode' in opts:
            m_index = opts['starting_mode']
        canvas.add_rect((15, 15), 12, 50*port.m - 30, fill="white")
        if m_index is not None:
            for i in range(port.m):
                canvas.add_text((21, 50 * i + 27), text=str(m_index+i), size=7, ta="middle")
        if port.name:
            canvas.add_text((27, 50*port.m - 9), text='[' + port.name + ']', size=6, ta="right", fontstyle="italic")
