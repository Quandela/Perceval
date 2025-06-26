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

from perceval.utils.logging import get_logger

from ..format import Format
from .renderer_interface import ICircuitRenderer
from .abstract_skin import ASkin
from .canvas_renderer import CanvasRenderer, PreRenderer
from .text_renderer import TextRenderer
import networkx as nx

import importlib

class RendererFactory:
    drawsvg_spec = importlib.util.find_spec("drawsvg")
    latexcodec_spec = importlib.util.find_spec("latexcodec")
    matplotlib_spec = importlib.util.find_spec("matplotlib")
    _CANVAS = {
        Format.LATEX: None,
        Format.HTML: None,
        Format.MPLOT: None,
    }
    _TOMOGRAPHY = None
    _GRAPH = None
    _DENSITY_MATRIX = None

    if drawsvg_spec:
        from ..canvas.svg_canvas import SvgCanvas
        _CANVAS[Format.HTML] = SvgCanvas
    if latexcodec_spec:
        from ..canvas.latex_canvas import LatexCanvas
        _CANVAS[Format.LATEX] = LatexCanvas
    if matplotlib_spec:
        from ..canvas.mplot_canvas import MplotCanvas
        from ..mplotlib_renderers.density_matrix_renderer import DensityMatrixRenderer
        from ..mplotlib_renderers.graph_renderer import GraphRenderer
        from ..mplotlib_renderers.tomography_renderer import TomographyRenderer
        _CANVAS[Format.MPLOT] = MplotCanvas
        _TOMOGRAPHY = TomographyRenderer
        _GRAPH = GraphRenderer
        _DENSITY_MATRIX = DensityMatrixRenderer

    @staticmethod
    def get_circuit_renderer(
        m: int,  # number of modes
        output_format: Format,  # rendering method
        skin: ASkin,  # skin
        **opts
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
        canvaType = RendererFactory._CANVAS[output_format]
        if not canvaType:
            get_logger().warn(f"""Missing dependencies to use {output_format}, defaulting to {Format.TEXT}
                              Use `pip install perceval-quandela[rendering]` to install needed packages""")
            return TextRenderer(m), None

        canvas = canvaType(**opts)
        return CanvasRenderer(m, canvas, skin), PreRenderer(m, skin)

    @staticmethod
    def get_tomography_renderer(
        output_format: Format,  # rendering method
        **opts):
        if output_format == Format.TEXT or output_format == Format.LATEX:
            raise TypeError(f"Tomography plot does not support {output_format}")
        if RendererFactory._TOMOGRAPHY:
            return RendererFactory._TOMOGRAPHY(**opts)

        raise Exception(f"""Missing dependencies to use {output_format}, defaulting to {Format.TEXT}
                            Use `pip install perceval-quandela[rendering]` to install needed packages""")

    @staticmethod
    def get_graph_renderer(
        output_format: Format,  # rendering method
        **opts):

        if output_format not in {Format.MPLOT, Format.LATEX}:
            raise TypeError(f"Graph plot does not support {output_format}")

        if output_format == Format.MPLOT:
            return RendererFactory._GRAPH(**opts)

        class GraphLatexRenderer:
            def render(g: nx.Graph):
                return nx.to_latex(g)
        return GraphLatexRenderer

    @staticmethod
    def get_density_matrix_renderer(
        output_format: Format,  # rendering method
        **opts):
        if output_format == Format.TEXT or output_format == Format.LATEX:
            raise TypeError(f"DensityMatrix plot does not support {output_format}")
        if RendererFactory._DENSITY_MATRIX:
            return RendererFactory._DENSITY_MATRIX(**opts)

        raise Exception("""Missing dependencies to use {}, defaulting to {}
                            Use `pip install perceval-quandela[rendering]` to install needed packages""".format(output_format, Format.TEXT))
