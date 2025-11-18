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
import copy
import os

from exqalibur import BSSamples
from multipledispatch import dispatch
import networkx as nx
import sympy as sp
from tabulate import tabulate

from perceval.algorithm import Analyzer, AProcessTomography
from perceval.components import (ACircuit, Circuit, AProcessor, Port, Herald, AFFConfigurator, Experiment,
                                 non_unitary_components as nl)
from .drawsvg_wrapper import DrawsvgWrapper
from .circuit.create_renderer import RendererFactory
from .circuit import DisplayConfig, ASkin
from perceval.utils import BasicState, Matrix, simple_float, simple_complex, DensityMatrix, mlstr, ModeType, Encoding
from perceval.utils.logging import get_logger, channel
from perceval.utils.states import StateVector, BSCount, BSDistribution, SVDistribution
from perceval.runtime import JobGroup

from .format import Format
from ._processor_utils import collect_herald_info


in_notebook = False


def in_ide():
    ide_detected = False
    for key in os.environ:
        if 'PYCHARM' in key or 'SPY_PYTHONPATH' in key or 'VSCODE' in key:
            ide_detected = True
    get_logger().debug(f"IDE detected: {ide_detected}", channel.general)
    return ide_detected


try:
    from IPython import get_ipython
    if 'IPKernelApp' in get_ipython().config:
        in_notebook = True
        from IPython.display import display, Math, HTML
except (ImportError, AttributeError):
    pass




def pdisplay_circuit(
        circuit: ACircuit,
        output_format: Format = Format.TEXT,
        recursive: bool = False,
        compact: bool = False,
        precision: float = 1e-6,
        nsimplify: bool = True,
        skin: ASkin = None,
        **opts):
    if skin is None:
        skin = DisplayConfig.get_selected_skin(compact_display=compact)
    skin.precision = precision
    skin.nsimplify = nsimplify
    w, h = skin.get_size(circuit, recursive)
    renderer, _ = RendererFactory.get_circuit_renderer(
        circuit.m,
        output_format=output_format,
        skin=skin,
        total_width=w,
        total_height=h,
        **opts)

    renderer.open()
    renderer.render_circuit(circuit, recursive=recursive, precision=precision, nsimplify=nsimplify)
    renderer.close()
    renderer.add_mode_index()
    return renderer.draw()


def pdisplay_processor(processor: AProcessor,
                       output_format: Format = Format.TEXT,
                       recursive: bool = False,
                       compact: bool = False,
                       precision: float = 1e-6,
                       nsimplify: bool = True,
                       skin: ASkin = None,
                       **opts):
    return pdisplay_experiment(processor.experiment, output_format, recursive, compact, precision, nsimplify, skin, **opts)


def pdisplay_experiment(processor: Experiment,
                       output_format: Format = Format.TEXT,
                       recursive: bool = False,
                       compact: bool = False,
                       precision: float = 1e-6,
                       nsimplify: bool = True,
                       skin: ASkin = None,
                       **opts):
    n_modes = processor.circuit_size
    if skin is None:
        skin = DisplayConfig.get_selected_skin(compact_display=compact)
    skin.precision = precision
    skin.nsimplify = nsimplify
    w, h = skin.get_size(processor, recursive)
    renderer, pre_renderer = RendererFactory.get_circuit_renderer(
        n_modes,
        output_format=output_format,
        skin=skin,
        total_width=w,
        total_height=h,
        compact=compact,
        **opts)

    herald_info = {}
    if len(processor.in_heralds):
        for k in processor.in_heralds.keys():
            renderer.set_mode_style(k, ModeType.HERALD)
        herald_info = collect_herald_info(processor, recursive)

    elif len(processor.heralds):
        herald_info = collect_herald_info(processor, recursive)

    original_mode_style = renderer._mode_style.copy()

    for rendering_pass in [pre_renderer, renderer]:
        if not rendering_pass:
            continue
        rendering_pass.set_herald_info(herald_info)
        rendering_pass.open()
        for r, c in processor.components:
            shift = r[0]
            if isinstance(c, ACircuit):
                c = Circuit(c.m).add(0, c)
            rendering_pass.render_circuit(
                c,
                recursive=recursive,
                precision=precision,
                nsimplify=nsimplify,
                shift=shift)
            if isinstance(c, AFFConfigurator):
                controlled_circuit = c.circuit_template()
                if isinstance(controlled_circuit, Circuit):
                    controlled_circuit = Circuit(controlled_circuit.m).add(0, controlled_circuit)
                rendering_pass.render_circuit(
                    controlled_circuit, recursive=recursive, shift=c.config_modes(r)[0]
                )

        rendering_pass.close()
        if pre_renderer:
            # Pass pre-computed subblock info to the main rendering pass.
            renderer.subblock_info.update(pre_renderer.subblock_info)

    in_ports_drawn_on_modes = []
    for port_range in processor._in_ports.values():
        # Avoids adding ports on heralded modes and modes with ports already defined
        in_ports_drawn_on_modes += port_range

    if isinstance(processor._input_state, BasicState):
        renderer.display_input_photons(processor._input_state, original_mode_style)
        # In this case add mono-mode ports on all modes containing none
        empty_raw_port = Port(Encoding.RAW, "")
        for i in range(processor.circuit_size):
            if i not in in_ports_drawn_on_modes:
                renderer.add_in_port(i, empty_raw_port)

    for port, port_range in processor._in_ports.items():
        renderer.add_in_port(port_range[0], port)

    renderer.add_detectors(processor._detectors)
    ports_drawn_on_modes = []
    for port, port_range in processor._out_ports.items():
        ports_drawn_on_modes += port_range
        if isinstance(port, Herald):
            det = processor._detectors[port_range[0]]
            if det is not None:
                port.detector_type = det.type
        renderer.add_out_port(port_range[0], port)
    for i in range(processor.circuit_size):
        if i not in ports_drawn_on_modes and \
                i not in processor.detectors_injected and processor._detectors[i] is not None:
            renderer.add_out_port(i, Port(Encoding.RAW, ""))

    renderer.add_mode_index(original_mode_style)
    return renderer.draw()


def pdisplay_matrix(matrix: Matrix, precision: float = 1e-6, output_format: Format = Format.TEXT) -> str:
    """
    :meta private:
    Generates representation of a matrix
    """

    def simp(value):
        if isinstance(value, (complex, float, int, sp.Number)) or (isinstance(value, sp.Expr) and len(value.free_symbols) == 0):
            return simple_complex(complex(value), precision=precision)[1]
        else:
            return value.__repr__()

    if output_format != Format.TEXT:
        marker = "$" if output_format == Format.HTML else ""
        if isinstance(matrix, sp.Matrix):
            return marker + sp.latex(matrix) + marker
        rows = []
        for j in range(matrix.shape[0]):
            row = []
            for v in matrix[j, :]:
                row.append(sp.S(simp(v)))
            rows.append(row)
        return marker + sp.latex(Matrix(rows, use_symbolic=True)) + marker
    if matrix.shape[0] == 1:
        return (mlstr("[") + mlstr("  ").join([simp(v) for v in matrix[0, :]]) + "]")._s
    else:
        s = mlstr("")
        for j in range(matrix.shape[1]):
            if j:
                s += "  "
            s += "\n".join([simp(v) for v in matrix[:, j]])
        h = s.height
        left_bracket = "⎡\n" + "⎢\n" * (h - 2) + "⎣"
        right_bracket = "⎤\n" + "⎥\n" * (h - 2) + "⎦"
        return (mlstr(left_bracket) + s + right_bracket)._s


_TABULATE_FMT_MAPPING = {
    Format.TEXT: 'pretty',
    Format.MPLOT: 'pretty',
    Format.HTML: 'html',
    Format.LATEX: 'latex'
}


def pdisplay_analyzer(analyzer: Analyzer, output_format: Format = Format.TEXT, nsimplify: bool = True,
                      precision: float = 1e-6):
    distribution = analyzer.distribution
    d = []
    for iidx, _ in enumerate(analyzer.input_states_list):
        d.append([simple_float(f.real, nsimplify=nsimplify, precision=precision)[1]
                  for f in list(distribution[iidx])])
    return tabulate(d, headers=[analyzer._mapping.get(o, str(o)) for o in analyzer.output_states_list],
                    showindex=[analyzer._mapping.get(i, str(i)) for i in analyzer.input_states_list],
                    tablefmt=_TABULATE_FMT_MAPPING[output_format])


def pdisplay_state_distrib(sv: StateVector | BSDistribution | SVDistribution | BSCount,
                           output_format: Format = Format.TEXT,
                           nsimplify: bool | None = None,
                           precision: float = 1e-6,
                           max_v: int | None = None,
                           sort: bool = True):
    """
    :meta private:
    Displays StateVector and ProbabilityDistribution as a table of state vs probability (probability amplitude in
    StateVector's case)
    """
    if nsimplify is None:
        # no numerical simplification by default if the number of displayed values is larger than 100
        nsimplify = False if (min(max_v or len(sv), len(sv)) > 100) else True
    if sort:
        the_keys = sorted(sv.keys(), key=lambda a: -abs(sv[a]))
    else:
        the_keys = list(sv.keys())
    d = []
    for k in the_keys[:max_v]:
        value = sv[k]
        if isinstance(value, float):
            value = simple_float(value, nsimplify=nsimplify, precision=precision)[1]
        elif isinstance(value, complex):
            values = []
            if value.real != 0:
                values.append(simple_float(value.real, nsimplify=nsimplify, precision=precision)[1])
            if value.imag != 0:
                values.append("I*" + simple_float(value.imag, nsimplify=nsimplify, precision=precision)[1])
            value = " + ".join(values) if values else "0"
        else:
            value = str(value)
        d.append([str(k), value])

    headers = ["state", "probability"]
    if isinstance(sv, StateVector):
        headers[1] = "prob. ampl."
    elif isinstance(sv, BSCount):
        headers[1] = "count"
    s_states = tabulate(d, headers=headers, tablefmt=_TABULATE_FMT_MAPPING[output_format])
    return s_states


def pdisplay_bs_samples(bs_samples: BSSamples, output_format: Format = Format.TEXT, max_v: int | None = 10):
    s_states = tabulate([[str(sample)] for sample in bs_samples[:max_v]],
                        headers=["states"], tablefmt=_TABULATE_FMT_MAPPING[output_format])
    return s_states


def pdisplay_tomography_chi(qpt: AProcessTomography, output_format: Format = Format.MPLOT, precision: float = 1E-6,
                            render_size=None, mplot_noshow: bool = False, mplot_savefig: str = None):
    renderer = RendererFactory.get_tomography_renderer(output_format, render_size=render_size, mplot_noshow=mplot_noshow, mplot_savefig=mplot_savefig)
    return renderer.render(qpt, precision=precision)


def pdisplay_density_matrix(dm,
                            output_format: Format = Format.MPLOT,
                            color: bool = True,
                            cmap='hsv',
                            mplot_noshow: bool = False,
                            mplot_savefig: str = None):
    """
    :meta private:
    :param dm:
    :param output_format:
    :param color: whether to display the phase according to some circular cmap
    :param cmap: the cmap to use fpr the phase indication
    """

    renderer = RendererFactory.get_density_matrix_renderer(output_format, color=color, cmap=cmap, mplot_noshow=mplot_noshow, mplot_savefig=mplot_savefig)
    return renderer.render(dm)


def pdisplay_graph(g: nx.Graph, output_format: Format = Format.MPLOT):
    renderer = RendererFactory.get_graph_renderer(output_format)
    return renderer.render(g)


def pdisplay_job_group(jg: JobGroup,  output_format: Format = Format.TEXT):
    progress = jg.progress()

    for key, value in progress.items():
        if isinstance(value, int):
            progress[key] = [value]

    return tabulate(progress.values(), headers=['Job Category', 'Count', 'Details'], showindex=progress.keys(),
                    tablefmt=_TABULATE_FMT_MAPPING[output_format])


@dispatch(object)
def _pdisplay(o, **kwargs):
    raise NotImplementedError(f"pdisplay not implemented for {type(o)}")

@dispatch(DensityMatrix)
def _pdisplay(dm, **kwargs):
    return pdisplay_density_matrix(dm, **kwargs)

@dispatch(AProcessTomography)
def _pdisplay(qpt, **kwargs):
    return pdisplay_tomography_chi(qpt, **kwargs)


@dispatch(JobGroup)
def _pdisplay(jg, **kwargs):
    return pdisplay_job_group(jg, **kwargs)


@dispatch((ACircuit, nl.TD, nl.LC))
def _pdisplay(circuit, **kwargs):
    return pdisplay_circuit(circuit, **kwargs)


@dispatch(AProcessor)
def _pdisplay(processor, **kwargs):
    return pdisplay_processor(processor, **kwargs)


@dispatch(Experiment)
def _pdisplay(experiment, **kwargs):
    return pdisplay_experiment(experiment, **kwargs)


@dispatch(Matrix)
def _pdisplay(matrix, **kwargs):
    return pdisplay_matrix(matrix, **kwargs)


@dispatch(Analyzer)
def _pdisplay(analyzer, **kwargs):
    return pdisplay_analyzer(analyzer, **kwargs)


@dispatch((StateVector, BSDistribution, SVDistribution))
def _pdisplay(distrib, **kwargs):
    # Work on a copy, in order to not force normalization simply because of a display call
    normalized_dist = copy.copy(distrib)
    normalized_dist.normalize()
    return pdisplay_state_distrib(normalized_dist, **kwargs)


@dispatch(BSCount)
def _pdisplay(bsc, **kwargs):
    return pdisplay_state_distrib(bsc, **kwargs)


@dispatch(BSSamples)
def _pdisplay(bssamples, **kwargs):
    return pdisplay_bs_samples(bssamples, **kwargs)


def _get_simple_number_kwargs(**kwargs):
    new_kwargs = {}
    keywords = ["precision", "nsimplify"]
    for kw in keywords:
        if kw in kwargs:
            new_kwargs[kw] = kwargs[kw]
    return new_kwargs

@dispatch((int,float))
def _pdisplay(f, **kwargs):
    return simple_float(f, **_get_simple_number_kwargs(**kwargs))[1]


@dispatch(complex)
def _pdisplay(c, **kwargs):
    return simple_complex(c, **_get_simple_number_kwargs(**kwargs))[1]

@dispatch(nx.Graph)
def _pdisplay(g, **kwargs):
    return pdisplay_graph(g, **kwargs)


def _default_output_format(o):
    """
    Deduces the best output format given the nature of the data to be displayed and the execution context
    """
    if in_notebook:
        if isinstance(o, Matrix):
            return Format.LATEX
        return Format.HTML
    elif in_ide() and (isinstance(o, (ACircuit, AProcessor, DensityMatrix, AProcessTomography, nl.TD, nl.LC, Experiment))):
        return Format.MPLOT
    return Format.TEXT


def pdisplay(o, output_format: Format = None, **opts):
    """ Pretty display
    Main rendering entry point. Several data types can be displayed using pdisplay.

    :param o: Perceval object to render
    :param output_format: Format controls where and how a figure is render (in a interactive window, the terminal, etc.)
        - MPLOT: Matplotlib drawing (default in IDE - spyder, pycharm or vscode)
        - HTML: HTML for data table, SVG for circuits/processors (default in notebook)
        - TEXT: Pretty text display (default in another cases)
        - LATEX: LaTex code, drawing with Tikz for circuits/processors

    opts:
        - skin (rendering.circuit.PhysSkin, SymbSkin or DebugSkin or any ASkin subclass instance):

            Skin controls how a circuit/processor is displayed:

                - PhysSkin(): physical skin (default),
                - DebugSkin(): Similar to PhysSkin but modes are bigger, ancillary modes are displayed,
                  components with variable parameters are red,
                - SymbSkin(): symbolic skin (thin black and white lines).

        - precision (float): numerical precision
        - nsimplify (bool): if True, tries to simplify numerical values by searching known values (pi, sqrt, fractions)
        - recursive (bool): if True, all hierarchy levels in a circuit/processor are displayed. Otherwise, only the top
          level is drawn, others are "black boxes"
        - max_v (int): Maximum number of displayed values in distributions
        - sort (bool): if True, sorts a distribution (descending order) before displaying
        - render_size: In SVG circuit/processor rendering, acts as a zoom factor (float)
          In Tomography display, is the size of the output plot in inches (tuple of two floats)
    """
    if output_format is None:
        output_format = _default_output_format(o)
        get_logger().debug(f"Output format defaulted to {output_format.name}", channel.general)
    res = _pdisplay(o, output_format=output_format, **opts)

    if res is None:
        return

    if isinstance(res, DrawsvgWrapper):
        return res.value
    elif in_notebook and output_format == Format.LATEX:
        display(Math(res))
    elif in_notebook and output_format == Format.HTML:
        display(HTML(res))
    else:
        print(res)


def pdisplay_to_file(o, path: str, output_format: Format = None, **opts):
    """
    Directly saves the result of pdisplay into a file without actually displaying it.

    :param o: Perceval object to render
    :param path: Path to file to save
    :param output_format: See :code:`pdisplay` for details.
      Contrarily to :code:`pdisplay`, this method always uses Format.MPLOT by default so you might need to specify it
      by hand for some kinds of objects.
    :param opts: See :code:`pdisplay` for details.
    """
    if output_format is None:
        output_format = Format.MPLOT
    if output_format == Format.MPLOT:
        opts['mplot_savefig'] = path
        opts['mplot_noshow'] = True
    res = _pdisplay(o, output_format=output_format, **opts)
    if res is None:
        raise RuntimeError(f"pdisplay_to_file not defined for type {type(o)}")

    if output_format == Format.MPLOT:
        return  # File was generated by the _pdisplay call

    if output_format == Format.TEXT:
        with open(path, 'w', encoding='utf-8') as f_out:
            f_out.write(res)
        return

    if output_format == Format.HTML:
        _, output_ext = os.path.splitext(path)
        try:
            if output_ext == ".png":
                res.save_png(path)  # May fail when rasterization is not available (i.e. on Windows)
            else:
                res.save_svg(path)
            return
        except Exception as e:
            get_logger().error(f"{e}", channel.general)

    if output_format == Format.LATEX:
        with open(path, 'w', encoding='utf-8') as f_out:
            f_out.write(res)
        return

    get_logger().warn(
        f"No output file could be created for {type(o)} object (format = {output_format.name}) at path {path}", channel.user)
