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

import os
import sys
import warnings
import random
import numpy as np

from .format import simple_float, simple_complex
import perceval as pcvl

in_notebook = False
in_pycharm_or_spyder = "PYCHARM_HOSTED" in os.environ or 'SPY_PYTHONPATH' in os.environ

global_params = {
    "min_p": 1e-16,
    "min_complex_component": 1e-6
}

try:
    from IPython import get_ipython
    if 'IPKernelApp' in get_ipython().config:
        in_notebook = True
        from IPython.display import HTML, display
except ImportError:
    pass
except AttributeError:
    pass


def _default_output_format(o):
    """
    Deduces the best output format given the nature of the data to be displayed and the execution context
    """
    if in_notebook:
        return "html"
    elif in_pycharm_or_spyder \
            and (isinstance(o, pcvl.ACircuit) or isinstance(o, pcvl.Processor)):
        return "mplot"
    return "text"


def pdisplay(o, output_format=None, to_file=None, **opts):
    if output_format is None:
        output_format = _default_output_format(o)

    if not hasattr(o, "pdisplay"):
        opts_simple = {}
        if "precision" in opts:
            opts_simple["precision"] = opts["precision"]
        if isinstance(o, (int, float)):
            r = simple_float(o, **opts_simple)[1]
        elif isinstance(o, complex):
            r = simple_complex(o, **opts_simple)[1]
        else:
            raise RuntimeError("pdisplay not defined for type %s" % type(o))
    else:
        r = o.pdisplay(output_format=output_format, **opts)

    if to_file:
        if 'drawSvg' in sys.modules:  # If drawSvg was imported beforehand
            import drawSvg
            if isinstance(r, drawSvg.Drawing):
                if output_format == "png":
                    r.savePng(to_file)
                else:
                    r.saveSvg(to_file)
                return
        else:
            warnings.warn("to_file parameter requires drawSvg to be installed on your system and manually imported.")

    if 'drawSvg' in sys.modules:  # If drawSvg was imported beforehand
        import drawSvg
        if isinstance(r, drawSvg.Drawing):
            return r
    elif in_notebook and output_format != "text":
        display(HTML(r))
    else:
        print(r)


def random_seed(param):
    """
    seed: int = None
    Initialize the seed used for random number generation

    :param seed: if None, use a time-based seed
    :return: the actual seed used
    """
    if param is not None:
        random.seed(param)
        np.random.seed(param)

    else:
        random.seed()
        np.random.seed()
