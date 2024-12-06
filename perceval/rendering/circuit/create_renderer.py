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

from .renderer_interface import ICircuitRenderer
from .abstract_skin import ASkin
from .canvas_renderer import CanvasRenderer, PreRenderer
from .text_renderer import TextRenderer
from ..canvas import LatexCanvas, MplotCanvas, SvgCanvas
from ..format import Format


_CANVAS = {
    Format.HTML: SvgCanvas,
    Format.MPLOT: MplotCanvas,
    Format.LATEX: LatexCanvas
}


def create_renderer(
    m: int,  # number of modes
    output_format: Format = Format.TEXT,  # rendering method
    skin: ASkin = None,  # skin (unused in text rendering)
    **opts,
) -> tuple[ICircuitRenderer, ICircuitRenderer | None]:
    """
    Creates a renderer given the selected format. Dispatches parameters to generated canvas objects
    A skin object is needed for circuit graphic rendering.

    This returns a (renderer, pre_renderer) tuple. It is recommended to
    invoke the pre-renderer on the circuit to correctly pre-compute
    additional position information that cannot be guessed in a single pass.
    """
    if output_format == Format.TEXT:
        return TextRenderer(m), None

    assert skin is not None, "A skin must be selected for circuit rendering"
    canvas = _CANVAS[output_format](**opts)
    return CanvasRenderer(m, canvas, skin), PreRenderer(m, skin)
