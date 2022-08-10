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

from perceval.components import Circuit, base_components as cp
from perceval.rendering.circuit.abstract_skin import ASkin
import sympy as sp


class PhysSkin(ASkin):
    def __init__(self, compact_display: bool = False):
        super().__init__({"stroke": "darkred", "stroke_width": 3}, compact_display)
        self.style_subcircuit = {"width": 2,
                        "fill": "lightpink",
                        "stroke_style": {"stroke": "darkred", "stroke_width": 1}}

    @dispatch((Circuit, cp.Unitary))
    def get_width(self, c) -> int:
        return c.m

    @dispatch((cp.SimpleBS, cp.GenericBS, cp.PBS))
    def get_width(self, c) -> int:
        return 2

    @dispatch((cp.PS, cp.TD, cp.PERM, cp.WP, cp.PR))
    def get_width(self, c) -> int:
        return 1

    @dispatch((cp.SimpleBS, cp.GenericBS))
    def get_shape(self, c):
        return self.bs_shape

    @dispatch(cp.PS)
    def get_shape(self, c):
        return self.ps_shape

    @dispatch(cp.PBS)
    def get_shape(self, c):
        return self.pbs_shape

    @dispatch(cp.TD)
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

    def bs_shape(self, circuit, canvas, content, **opts):
        split_content = content.split("\n")
        head_content = "\n".join([s for s in split_content
                                  if s.startswith("R=") or s.startswith("theta=")])
        bottom_content_list = [s for s in split_content
                               if not s.startswith("R=") and not s.startswith("theta=")]
        bottom_nline = len(bottom_content_list)
        bottom_size = 7 if bottom_nline < 3 else 6
        canvas.add_mline([0, 25, 28, 25, 47, 44], stroke_linejoin="round", **self.stroke_style)
        canvas.add_mline([53, 44, 72, 25, 100, 25], stroke_linejoin="round", **self.stroke_style)
        canvas.add_mline([0, 75, 28, 75, 47, 56], stroke_linejoin="round", **self.stroke_style)
        canvas.add_mline([53, 56, 72, 75, 100, 75], stroke_linejoin="round", **self.stroke_style)
        canvas.add_rect((25, 43), 50, 14, fill="black")
        canvas.add_text((50, 80+5*bottom_nline), '\n'.join(bottom_content_list).replace('phi_', 'Φ_'),
                        size=bottom_size, ta="middle")
        canvas.add_text((50, 26), head_content.replace('theta=', 'Θ='), size=7, ta="middle")
        # Choose the side of the gray rectangle in beam splitter representation
        m = None
        if hasattr(circuit, '_phi_b') and circuit._phi_b.defined:  # GenericBS
            m = round(abs(float(circuit._phi_b.spv/sp.pi)))
        elif hasattr(circuit, '_phi') and circuit._phi.defined:  # SimpleBS
            m = round(abs(float(circuit._phi.spv/sp.pi)))
        if m is not None:
            if (m + 1) % 2:
                canvas.add_rect((25, 43), 50, 4, fill="lightgray")
            else:
                canvas.add_rect((25, 53), 50, 4, fill="lightgray")

    def ps_shape(self, circuit, canvas, content, **opts):
        canvas.add_mline([0, 25, 50, 25], **self.stroke_style)
        canvas.add_polygon([5, 40, 14, 40, 28, 10, 19, 10, 5, 40, 14, 40],
                           stroke="black", fill="gray", stroke_width=1, stroke_linejoin="miter")
        canvas.add_text((22, 38), text=content.replace("phi=", "Φ="), size=7, ta="left")

    def pbs_shape(self, circuit, canvas, content, **opts):
        canvas.add_mline([0, 25, 28, 25, 37.5, 37.5], **self.stroke_style, stroke_linejoin="round")
        canvas.add_mline([62.5, 37.5, 72, 25, 100, 25], **self.stroke_style, stroke_linejoin="round")
        canvas.add_mline([0, 75, 28, 75, 37.5, 62.5], **self.stroke_style, stroke_linejoin="round")
        canvas.add_mline([62.5, 62.5, 72, 75, 100, 75], **self.stroke_style, stroke_linejoin="round")
        canvas.add_mline([62.5, 62.5, 72, 75, 100, 75], **self.stroke_style, stroke_linejoin="round")
        canvas.add_polygon([25, 50, 50, 24, 75, 50, 50, 76, 25, 50], stroke="black", stroke_width=1, fill="gray")
        canvas.add_mline([25, 50, 75, 50], stroke="black", stroke_width=1)
        canvas.add_text((50, 86), text=content, size=7, ta="middle")

    def td_shape(self, circuit, canvas, content, **opts):
        canvas.add_circle((34, 14), 11, stroke_width=5, fill=None, stroke="white")
        canvas.add_circle((34, 14), 11, fill=None, **self.stroke_style)
        canvas.add_circle((25, 14), 11, stroke_width=5, fill=None, stroke="white")
        canvas.add_circle((25, 14), 11, fill=None, **self.stroke_style)
        canvas.add_circle((16, 14), 11, stroke_width=5, fill=None, stroke="white")
        canvas.add_circle((16, 14), 11, fill=None, **self.stroke_style)
        canvas.add_mline([0, 25, 19, 25], stroke="white", stroke_width=5)
        canvas.add_mline([0, 25, 19, 25], **self.stroke_style)
        canvas.add_mline([34, 25, 50, 25], stroke="white", stroke_width=5)
        canvas.add_mline([32, 25, 50, 25], **self.stroke_style)
        canvas.add_text((25, 38), content, 7, "middle")

    def unitary_shape(self, circuit, canvas, content, **opts):
        m = circuit.m
        for i in range(m):
            canvas.add_mpath(["M", 0, 25 + i*50, "l", 50*m, 0], **self.stroke_style)
        canvas.add_rect((5, 5), 50*m-10, 50*m-10, fill="gold")
        canvas.add_text((25*m, 25*m), size=10, ta="middle", text=circuit.name)

    def perm_shape(self, circuit, canvas, content, **opts):
        for an_input, an_output in enumerate(circuit.perm_vector):
            canvas.add_mline([3, 25+an_input*50, 47, 25+an_output*50],
                             stroke="white", stroke_width=6)
            canvas.add_mline([0, 25+an_input*50, 3, 25+an_input*50, 47, 25+an_output*50, 50, 25+an_output*50],
                             **self.stroke_style)

    def wp_shape(self, circuit, canvas, content, **opts):
        params = content.replace("xsi=", "ξ=").replace("delta=", "δ=").split("\n")
        canvas.add_mline([0, 25, 50, 25], **self.stroke_style)
        canvas.add_rect((13, 7), width=14, height=36, fill="gray",
                        stroke_width=1, stroke="black", stroke_linejoin="miter")
        canvas.add_mline([20, 7, 20, 43], stroke="black", stroke_width=1)
        canvas.add_text((28.5, 36), text=params[0], size=7, ta="left")
        canvas.add_text((28.5, 45), text=params[1], size=7, ta="left")

    def pr_shape(self, circuit, canvas, content, **opts):
        canvas.add_mline([0, 25, 15, 25], **self.stroke_style)
        canvas.add_mline([35, 25, 50, 25], **self.stroke_style)
        canvas.add_rect((14, 14), width=22, height=22, stroke="black", fill="lightgray",
                        stroke_width=1, stroke_linejoin="miter")
        canvas.add_mpath(["M", 18, 27, "c", 0.107, 0.131, 0.280, 0.131, 0.387, 0,
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
                          "h", -1.201, "c", -0.169, 0, -0.219, 0.106, -0.112, 0.237,
                          "z"
                          ], fill="black")
        canvas.add_text((25, 45), text=content.replace("delta=", "δ="), size=7, ta="middle")

    def subcircuit_shape(self, circuit, canvas, content, **opts):
        w = self.style_subcircuit['width']
        for idx in range(circuit.m):
            canvas.add_mline([0, 50*idx+25, w*50, 50*idx+25], **self.stroke_style)
        canvas.add_rect((2.5, 2.5), w*50 - 5, 50*circuit.m - 5,
                        fill=self.style_subcircuit['fill'], **self.style_subcircuit['stroke_style'])
        canvas.add_text((16, 16), content.upper(), 8)

    def source_shape(self, circuit, canvas, content, **opts):
        r = 10
        color = "lightgray"
        if 'color' in opts:
            color = opts['color']
        canvas.add_mpath(["M", 0, 25, "c", 0, 0, 0, -r, r, -r,
                          "h", 8, "v", 2 * r, "h", -8,
                          "c", -r, 0, -r, -r, -r, -r, "z"],
                         stroke="black", stroke_width=1, fill=color)
        if 'name' in opts and opts['name']:
            canvas.add_text((8, 44), text='[' + opts['name'] + ']', size=6, ta="middle", fontstyle="italic")
        if content:
            canvas.add_text((10, 28), text=content, size=7, ta="middle")

    def detector_shape(self, circuit, canvas, content, **opts):
        r = 10  # Radius of the half-circle
        color = "lightgray"
        if 'color' in opts:
            color = opts['color']
        canvas.add_mpath(["M", 20, 35, "h", -8, "v", -2 * r, "h", 8,
                          "c", 0, 0, r, 0, r, r,
                          "c", 0, r, -r, r, -r, r, "z"],
                         stroke="black", stroke_width=1, fill=color)
        if 'name' in opts:
            canvas.add_text((18, 44), text='[' + opts['name'] + ']', size=6, ta="middle", fontstyle="italic")
        if content:
            canvas.add_text((20, 28), text=content, size=7, ta="middle")
