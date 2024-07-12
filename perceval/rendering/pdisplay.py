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
import copy
import os
import numpy

from multipledispatch import dispatch
import sympy as sp
from tabulate import tabulate
from typing import Union
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings(
        action='ignore',
        category=RuntimeWarning)
    import drawsvg

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D

from perceval.algorithm.analyzer import Analyzer
from perceval.algorithm import AProcessTomography
from perceval.components import ACircuit, Circuit, AProcessor, non_unitary_components as nl
from perceval.rendering.circuit import DisplayConfig, create_renderer, ModeStyle
from perceval.rendering._density_matrix_utils import _csr_to_rgb, _csr_to_greyscale, generate_ticks, _complex_to_rgb
from perceval.utils.format import simple_float, simple_complex
from perceval.utils.matrix import Matrix
from perceval.utils import DensityMatrix
from perceval.utils.mlstr import mlstr
from perceval.utils.statevector import ProbabilityDistribution, StateVector, BSCount
from .format import Format
from ._processor_utils import collect_herald_info

import math


in_notebook = False


def in_ide():
    for key in os.environ:
        if 'PYCHARM' in key or 'SPY_PYTHONPATH' in key or 'VSCODE' in key:
            return True
    return False


try:
    from IPython import get_ipython
    if 'IPKernelApp' in get_ipython().config:
        in_notebook = True
        from IPython.display import display, Math, HTML
except (ImportError, AttributeError):
    pass


def pdisplay_circuit(
        circuit: ACircuit,
        map_param_kid: dict = None,
        output_format: Format = Format.TEXT,
        recursive: bool = False,
        compact: bool = False,
        precision: float = 1e-6,
        nsimplify: bool = True,
        skin=None,
        **opts):
    if skin is None:
        skin = DisplayConfig.get_selected_skin(compact_display=compact)
    w, h = skin.get_size(circuit, recursive)
    renderer, _ = create_renderer(
        circuit.m,
        output_format=output_format,
        skin=skin,
        total_width=w,
        total_height=h,
        **opts)

    if map_param_kid is None:
        map_param_kid = circuit.map_parameters()
    renderer.open()
    renderer.render_circuit(circuit, map_param_kid, recursive=recursive, precision=precision, nsimplify=nsimplify)
    renderer.close()
    renderer.add_mode_index()
    return renderer.draw()


def pdisplay_processor(processor: AProcessor,
                       output_format: Format = Format.TEXT,
                       recursive: bool = False,
                       compact: bool = False,
                       precision: float = 1e-6,
                       nsimplify: bool = True,
                       skin=None,
                       **opts):
    n_modes = processor.circuit_size
    if skin is None:
        skin = DisplayConfig.get_selected_skin(compact_display=compact)
    w, h = skin.get_size(processor, recursive)
    renderer, pre_renderer = create_renderer(
        n_modes,
        output_format=output_format,
        skin=skin,
        total_width=w,
        total_height=h,
        compact=compact,
        **opts)

    herald_info = {}
    if len(processor.heralds):
        for k in processor.heralds.keys():
            renderer.set_mode_style(k, ModeStyle.HERALD)
        herald_info = collect_herald_info(processor, recursive)

    for rendering_pass in [pre_renderer, renderer]:
        if not rendering_pass:
            continue
        rendering_pass.set_herald_info(herald_info)
        rendering_pass.open()
        for r, c in processor.components:
            shift = r[0]
            if isinstance(c, Circuit):
                c = Circuit(c.m).add(0, c)
            rendering_pass.render_circuit(
                c,
                recursive=recursive,
                precision=precision,
                nsimplify=nsimplify,
                shift=shift)
        rendering_pass.close()
        if pre_renderer:
            # Pass pre-computed subblock info to the main rendering pass.
            renderer.subblock_info.update(pre_renderer.subblock_info)

    for port, port_range in processor._in_ports.items():
        renderer.add_in_port(port_range[0], port)

    for port, port_range in processor._out_ports.items():
        renderer.add_out_port(port_range[0], port)
    return renderer.draw()


def pdisplay_matrix(matrix: Matrix, precision: float = 1e-6, output_format: Format = Format.TEXT) -> str:
    """
    :meta private:
    Generates representation of a matrix
    """

    def simp(value):
        if isinstance(value, complex) or isinstance(value, int) or isinstance(value, float) or\
           isinstance(value, sp.Number) or (isinstance(value, sp.Expr) and len(value.free_symbols) == 0):
            return simple_complex(complex(value), precision=precision)[1]
        else:
            return value.__repr__()

    if output_format != Format.TEXT:
        marker = output_format == Format.HTML and "$" or ""
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
        d.append([simple_float(f, nsimplify=nsimplify, precision=precision)[1]
                  for f in list(distribution[iidx])])
    return tabulate(d, headers=[analyzer._mapping.get(o, str(o)) for o in analyzer.output_states_list],
                    showindex=[analyzer._mapping.get(i, str(i)) for i in analyzer.input_states_list],
                    tablefmt=_TABULATE_FMT_MAPPING[output_format])


def pdisplay_state_distrib(sv: Union[StateVector, ProbabilityDistribution, BSCount],
                           output_format: Format = Format.TEXT, nsimplify=True, precision=1e-6, max_v=None, sort=True):
    """
    :meta private:
    Displays StateVector and ProbabilityDistribution as a table of state vs probability (probability amplitude in
    StateVector's case)
    """
    if sort:
        the_keys = sorted(sv.keys(), key=lambda a: -abs(sv[a]))
    else:
        the_keys = list(sv.keys())
    if max_v is not None:
        the_keys = the_keys[:max_v]
    d = []
    for k in the_keys:
        value = sv[k]
        if isinstance(value, float):
            value = simple_float(value, nsimplify=nsimplify, precision=precision)[1]
        elif isinstance(value, complex):
            real_part = imag_part = ""
            if value.real != 0:
                real_part = simple_float(value.real, nsimplify=nsimplify, precision=precision)[1]
            if value.imag != 0:
                imag_part = "I*" + simple_float(value.imag, nsimplify=nsimplify, precision=precision)[1]
            value = real_part + imag_part
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

    # labels on x- and y- axes


def _generate_pauli_captions(nqubit: int):
    from perceval.algorithm.tomography.tomography_utils import _generate_pauli_index
    pauli_indices = _generate_pauli_index(nqubit)
    pauli_names = []
    for subset in pauli_indices:
        pauli_names.append([member.name for member in subset])

    basis = []
    for val in pauli_names:
        basis.append(''.join(val))
    return basis


def _get_sub_figure(ax: Axes3D, array: numpy.array, basis_name: list):
    # Data
    size = array.shape[0]
    x = numpy.array([[i] * size for i in range(size)]).ravel()  # x coordinates of each bar
    y = numpy.array([i for i in range(size)] * size)  # y coordinates of each bar
    z = numpy.zeros(size * size)  # z coordinates of each bar
    dxy = numpy.ones(size * size) * 0.5  # Width/Lenght of each bar
    dz = array.ravel()  # length along z-axis of each bar (height)

    # Colors
    # get range of colorbars so we can normalize
    max_height = numpy.max(dz)
    min_height = numpy.min(dz)
    color_map = plt.get_cmap('viridis_r')
    if max_height != min_height:
        has_only_one_value = False
        # scale each z to [0,1], and get their rgb values
        rgba = [color_map((k - min_height) / max_height) for k in dz]
    else:
        has_only_one_value = True
        rgba = [color_map(0)]

    # Caption
    font_size = 6

    # XY
    ax.set_xticks(numpy.arange(size) + 1)
    ax.set_yticks(numpy.arange(size) + 1)
    ax.tick_params(axis='x', which='major', labelsize=font_size)
    ax.set_xticklabels(basis_name)
    ax.tick_params(axis='y', which='major', labelsize=font_size)
    ax.set_yticklabels(basis_name)

    # Z
    if not has_only_one_value:
        ax.set_zlim(zmin=dz.min(), zmax=dz.max())
    ax.tick_params('z', which='both', labelsize=font_size)
    ax.grid(True, axis='z', which='major', linewidth=2)
    # interval = [v for v in ax.get_zticks() if v > 0][0]
    # ax.zaxis.set_minor_locator(ticker.MultipleLocator(interval/5))

    # Plot
    ax.bar3d(x, y, z, dxy, dxy, dz, color=rgba, alpha=0.7)
    ax.view_init(elev=30, azim=45)


def pdisplay_tomography_chi(qpt: AProcessTomography, output_format: Format = Format.MPLOT, precision: float = 1E-6,
                            render_size=None, mplot_noshow: bool = False, mplot_savefig: str = None):
    if output_format == Format.TEXT or output_format == Format.LATEX:
        raise TypeError(f"Tomography plot does not support {output_format}")

    chi_op = qpt.chi_matrix()

    if render_size is not None and isinstance(render_size, tuple) and len(render_size) == 2:
        fig = plt.figure(figsize=render_size)
    else:
        fig = plt.figure()
    pauli_captions = _generate_pauli_captions(qpt._nqubit)
    significant_digit = int(math.log10(1 / precision))

    # Real plot
    ax = fig.add_subplot(121, projection='3d')
    ax.set_title("Re[$\\chi$]")
    real_chi = numpy.round(chi_op.real, significant_digit)
    _get_sub_figure(ax, real_chi, pauli_captions)

    # Imag plot
    ax = fig.add_subplot(122, projection='3d')
    ax.set_title("Im[$\\chi$]")
    imag_chi = numpy.round(chi_op.imag, significant_digit)
    _get_sub_figure(ax, imag_chi, pauli_captions)

    if not mplot_noshow:
        plt.show()
    if mplot_savefig:
        fig.savefig(mplot_savefig, bbox_inches="tight", format="svg")
        return ""

    return None


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

    if output_format == Format.TEXT or output_format == Format.LATEX:
        raise TypeError(f"DensityMatrix plot does not support {output_format}")
    fig = plt.figure()

    if color:
        img = _csr_to_rgb(dm.mat, cmap)
        plt.imshow(img)
    else:
        img = _csr_to_greyscale(dm.mat)
        plt.imshow(img, cmap='gray')

    l1, l2 = generate_ticks(dm)

    plt.yticks(l1, l2)
    plt.xticks([])

    if not mplot_noshow:
        plt.show()
    if mplot_savefig:
        fig.savefig(mplot_savefig, bbox_inches="tight", format="svg")
        return ""


@dispatch(object)
def _pdisplay(o, **kwargs):
    raise NotImplementedError(f"pdisplay not implemented for {type(o)}")

@dispatch(DensityMatrix)
def _pdisplay(dm, **kwargs):
    return pdisplay_density_matrix(dm, **kwargs)

@dispatch(AProcessTomography)
def _pdisplay(qpt, **kwargs):
    return pdisplay_tomography_chi(qpt, **kwargs)


@dispatch((ACircuit, nl.TD))
def _pdisplay(circuit, **kwargs):
    return pdisplay_circuit(circuit, **kwargs)


@dispatch(AProcessor)
def _pdisplay(processor, **kwargs):
    return pdisplay_processor(processor, **kwargs)


@dispatch(Matrix)
def _pdisplay(matrix, **kwargs):
    return pdisplay_matrix(matrix, **kwargs)


@dispatch(Analyzer)
def _pdisplay(analyzer, **kwargs):
    return pdisplay_analyzer(analyzer, **kwargs)


@dispatch((StateVector, ProbabilityDistribution))
def _pdisplay(distrib, **kwargs):
    # Work on a copy, in order to not force normalization simply because of a display call
    normalized_dist = copy.copy(distrib)
    normalized_dist.normalize()
    return pdisplay_state_distrib(normalized_dist, **kwargs)


@dispatch(BSCount)
def _pdisplay(bsc, **kwargs):
    return pdisplay_state_distrib(bsc, **kwargs)


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


def _default_output_format(o):
    """
    Deduces the best output format given the nature of the data to be displayed and the execution context
    """
    if in_notebook:
        if isinstance(o, Matrix):
            return Format.LATEX
        return Format.HTML
    elif in_ide() and (isinstance(o, (ACircuit, AProcessor, DensityMatrix, AProcessTomography))):
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
            Skin controls how a circuit/processor is displayed
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
    res = _pdisplay(o, output_format=output_format, **opts)

    if res is None:
        return

    if isinstance(res, drawsvg.Drawing):
        return res
    elif in_notebook and output_format == Format.LATEX:
        display(Math(res))
    elif in_notebook and output_format == Format.HTML:
        display(HTML(res))
    else:
        print(res)


def pdisplay_to_file(o, path: str, output_format: Format = None, **opts):
    if output_format is None:
        output_format = Format.MPLOT
    if output_format == Format.MPLOT:
        opts['mplot_savefig'] = path
        opts['mplot_noshow'] = True
    res = _pdisplay(o, output_format=output_format, **opts)
    if res is None:
        raise RuntimeError("pdisplay_to_file not defined for type %s" % type(o))

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
        except:
            pass

    if output_format == Format.LATEX:
        with open(path, 'w', encoding='utf-8') as f_out:
            f_out.write(res)
        return

    warnings.warn(
        f"No output file could be created for {type(o)} object (format = {output_format.name}) at path {path}")
