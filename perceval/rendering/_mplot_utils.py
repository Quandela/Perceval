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

import matplotlib
import os
import platform


def autoselect_backend():
    try:
        # The next line may raise an exception if the backend needs to be autodetected, because gtk* candidate backends
        # require cairo which is not available on Windows
        matplotlib.rcParams['backend']
    except Exception:  # We cannot guess the exception type we need to catch here: it can come from any Matplotlib
        # backend or third party. We do not have control over this code

        # In order to avoid matplotlib trying to use cairo (which is a dependency of cairocffi retrieved by drawSvg),
        # hint the backend given the execution context, and avoid cairo related backends at all cost!
        in_notebook = False
        in_pycharm_or_spyder = "PYCHARM_HOSTED" in os.environ or 'SPY_PYTHONPATH' in os.environ

        try:
            from IPython import get_ipython
            in_notebook = 'IPKernelApp' in get_ipython().config
        except (ImportError, AttributeError):
            pass

        if in_pycharm_or_spyder:
            matplotlib.use("module://backend_interagg")
        elif in_notebook:
            matplotlib.use("module://matplotlib_inline.backend_inline")
        elif platform.system() == "Darwin":
            matplotlib.use("MacOSX")
        else:
            try:
                import tkinter
                matplotlib.use("TkAgg")
            except (ModuleNotFoundError, ImportError):
                # Last chance: use "agg" non-interactive backend (which should work "anywhere").
                matplotlib.use("agg")
