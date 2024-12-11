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
from perceval.components import unitary_components as cp
from .phys_skin import PhysSkin, ModeType


class DebugSkin(PhysSkin):
    def __init__(self, compact_display: bool = False):
        super().__init__(compact_display)
        self.style[ModeType.PHOTONIC]["stroke_width"] = 8
        self.style[ModeType.HERALD] = {"stroke": "orange", "stroke_width": 3}  # Display ancillary modes in yellow

    def ps_shape(self, circuit: cp.PS, canvas, mode_style):
        canvas.add_mline([0, 25, 50, 25], **self.style[ModeType.PHOTONIC])
        fill_color = "gray" if circuit.defined else "red"
        canvas.add_polygon([5, 40, 14, 40, 28, 10, 19, 10, 5, 40, 14, 40],
                           stroke="black", fill=fill_color, stroke_width=1, stroke_linejoin="miter")
        canvas.add_text((22, 38), text=self._get_display_content(circuit).replace("phi=", "Φ="), size=7, ta="left")

    def barrier_shape(self, circuit, canvas, mode_style):
        m = circuit.m
        if not circuit.visible:
            # even if invisible, draw a thin line for debug purpose
            canvas.add_rect((0, 10), 2, 50 * m - 20, fill="whitesmoke", stroke="whitesmoke")
            return

        if canvas.background_color is None:
            canvas.add_rect((10, 10), 30, 50 * m - 20, fill="whitesmoke", stroke="whitesmoke")
        for i in range(m):
            style = self.style[mode_style[i]]
            if style["stroke"]:
                canvas.add_mpath(["M", 0, 25 + i*50, "l", 50, 0], **style)
        canvas.add_rect((24, 10), 2, 50 * m - 20, fill="dimgrey", stroke="dimgrey")
