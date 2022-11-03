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
import copy
import os
from multipledispatch import dispatch
import sympy as sp
from tabulate import tabulate
from typing import Union
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings(
        action='ignore',
        category=RuntimeWarning)
    import drawSvg

from perceval.algorithm.analyzer import Analyzer
from perceval.components import ACircuit, Circuit, Processor, non_unitary_components as nl
from perceval.rendering.circuit import DisplayConfig, create_renderer, ModeStyle
from perceval.utils.format import simple_float, simple_complex
from perceval.utils.matrix import Matrix
from perceval.utils.mlstr import mlstr
from perceval.utils.statevector import ProbabilityDistribution, StateVector, SVDistribution
from .format import Format
from ._processor_utils import precompute_herald_pos


in_notebook = False
in_pycharm_or_spyder = "PYCHARM_HOSTED" in os.environ or 'SPY_PYTHONPATH' in os.environ

try:
    from IPython import get_ipython
    if 'IPKernelApp' in get_ipython().config:
        in_notebook = True
        from IPython.display import HTML, display
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
    renderer = create_renderer(circuit.m, output_format=output_format, skin=skin,
                               total_width=w, total_height=h, **opts)
    if map_param_kid is None:
        map_param_kid = circuit.map_parameters()
    renderer.open()
    renderer.render_circuit(circuit, map_param_kid, recursive=recursive, precision=precision, nsimplify=nsimplify)
    renderer.close()
    renderer.add_mode_index()
    return renderer.draw()


def pdisplay_processor(processor: Processor,
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
    renderer = create_renderer(n_modes, output_format=output_format, skin=skin,
                               total_width=w, total_height=h, compact=compact, **opts)
    if len(processor.heralds):
        for k in processor.heralds.keys():
            renderer.set_mode_style(k, ModeStyle.HERALD)
        if recursive:
            out_herald_info = precompute_herald_pos(processor)
            renderer.set_out_herald_info(out_herald_info)
    renderer.open()
    for r, c in processor.components:
        shift = r[0]
        if isinstance(c, Circuit):
            c = Circuit(c.m).add(0, c)
        renderer.render_circuit(c,
                                recursive=recursive,
                                precision=precision,
                                nsimplify=nsimplify,
                                shift=shift)
    renderer.close()

    for port, port_range in processor._in_ports.items():
        renderer.add_in_port(port_range[0], port)

    for port, port_range in processor._out_ports.items():
        renderer.add_out_port(port_range[0], port)
    return renderer.draw()


def pdisplay_matrix(matrix: Matrix, precision: float = 1e-6, output_format: Format = Format.TEXT) -> str:
    """
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
            return marker+sp.latex(matrix)+marker
        rows = []
        for j in range(matrix.shape[0]):
            row = []
            for v in matrix[j, :]:
                row.append(sp.S(simp(v)))
            rows.append(row)
        return marker+sp.latex(Matrix(rows, use_symbolic=True))+marker
    if matrix.shape[0] == 1:
        return (mlstr("[")+mlstr("  ").join([simp(v) for v in matrix[0, :]])+"]")._s
    else:
        s = mlstr("")
        for j in range(matrix.shape[1]):
            if j:
                s += "  "
            s += "\n".join([simp(v) for v in matrix[:, j]])
        h = s.height
        left_bracket = "⎡\n"+"⎢\n"*(h-2)+"⎣"
        right_bracket = "⎤\n"+"⎥\n"*(h-2)+"⎦"
        return (mlstr(left_bracket)+s+right_bracket)._s


_TABULATE_FMT_MAPPING = {
    Format.TEXT: 'pretty',
    Format.MPLOT: 'pretty',
    Format.HTML: 'html',
    Format.LATEX: 'latex'
}


def pdisplay_analyzer(analyser: Analyzer, output_format: Format = Format.TEXT, nsimplify: bool = True,
                      precision: float = 1e-6):
    distribution = analyser.distribution
    d = []
    for iidx, _ in enumerate(analyser.input_states_list):
        d.append([simple_float(f, nsimplify=nsimplify, precision=precision)[1]
                  for f in list(distribution[iidx])])
    return tabulate(d, headers=[analyser._mapping.get(o, str(o)) for o in analyser.output_states_list],
                    showindex=[analyser._mapping.get(i, str(i)) for i in analyser.input_states_list],
                    tablefmt=_TABULATE_FMT_MAPPING[output_format])


def pdisplay_state_distrib(sv: Union[StateVector, ProbabilityDistribution], output_format: Format = Format.TEXT,
                           nsimplify=True, precision=1e-6, max_v=None, sort=True):
    """
    Displays StateVector and ProbabilityDistribution as a table of state vs probability (probability amplitude in
    StateVector's case)
    """
    sv = copy.copy(sv)  # Work on a copy, in order to not force normalization simply because of a display call
    sv.normalize()
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
        d.append([k, value])

    headers = ["state", "probability"]
    if isinstance(sv, StateVector):
        headers[1] = "prob. ampl."
    s_states = tabulate(d, headers=headers, tablefmt=_TABULATE_FMT_MAPPING[output_format])
    return s_states


@dispatch(object)
def _pdisplay(_, **kwargs):
    return None


@dispatch((ACircuit, nl.TD))
def _pdisplay(circuit, **kwargs):
    return pdisplay_circuit(circuit, **kwargs)


@dispatch(Processor)
def _pdisplay(processor, **kwargs):
    return pdisplay_processor(processor, **kwargs)


@dispatch(Matrix)
def _pdisplay(matrix, **kwargs):
    return pdisplay_matrix(matrix, **kwargs)


@dispatch(Analyzer)
def _pdisplay(analyzer, **kwargs):
    return pdisplay_analyzer(analyzer, **kwargs)


@dispatch((StateVector, ProbabilityDistribution))
def _pdisplay(statevector, **kwargs):
    return pdisplay_state_distrib(statevector, **kwargs)


def _default_output_format(o):
    """
    Deduces the best output format given the nature of the data to be displayed and the execution context
    """
    if in_notebook:
        return Format.HTML
    elif in_pycharm_or_spyder and (isinstance(o, ACircuit) or isinstance(o, Processor)):
        return Format.MPLOT
    return Format.TEXT


def pdisplay(o, output_format: Format = None, **opts):
    """
    Main rendering entry point. Several data types can be displayed using pdisplay.
    """
    if output_format is None:
        output_format = _default_output_format(o)
    res = _pdisplay(o, output_format=output_format, **opts)
    if res is None:
        opts_simple = {}
        if "precision" in opts:
            opts_simple["precision"] = opts["precision"]
        if isinstance(o, (int, float)):
            res = simple_float(o, **opts_simple)[1]
        elif isinstance(o, complex):
            res = simple_complex(o, **opts_simple)[1]
        else:
            raise RuntimeError("pdisplay not defined for type %s" % type(o))

    if isinstance(res, drawSvg.Drawing):
        return res
    elif in_notebook and output_format != Format.TEXT:
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
                res.savePng(path)  # May fail when rasterization is not available (i.e. on Windows)
            else:
                res.saveSvg(path)
            return
        except:
            pass

    warnings.warn(
        f"No output file could be created for {type(o)} object (format = {output_format.name}) at path {path}")
