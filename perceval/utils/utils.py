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
from .format import simple_float, simple_complex
in_notebook = False
in_pycharm_or_spyder = "PYCHARM_HOSTED" in os.environ or 'SPY_PYTHONPATH' in os.environ

global_params = {
    "min_p": 1e-16,
    "min_complex_component": 1e-8
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


def pdisplay(o, output_format=None, **opts):
    if output_format is None:
        if in_notebook:
            output_format = "html"
        elif hasattr(o, "delay_circuit") and in_pycharm_or_spyder:
            # characterize ACCircuit objects
            output_format = "mplot"
        else:
            output_format = "text"

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
    if in_notebook and output_format != "text":
        display(HTML(r))
    else:
        print(r)
