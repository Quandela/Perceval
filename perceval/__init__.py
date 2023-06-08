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

"""
Through a simple object-oriented python API, Perceval provides tools for building a circuit with linear optics
components, defining single-photon sources and their error model, manipulating Fock states, running simulations,
reproducing published experimental papers results and experimenting a new generation of quantum algorithms.

It is interfaced with the available QPUs on https://cloud.quandela.com, so it is possible to run computations on an
actual photonic computer.

Perceval aims to be a companion tool for developing discrete-variable photonic circuits
    - while simulating their design, modeling their ideal and real-life behaviour;
    - and proposing a normalized interface to control photonic quantum computers;
    - while using powerful simulation backends to get state-of-the-art simulation;
    - and also allowing direct access to the QPUs of Quandela.

See also:
    - Perceval user documentation: https://perceval.quandela.net/docs/
    - Quandela cloud documentation: https://cloud.quandela.com/webide/documentation (requires a free account to access)
"""

from pkg_resources import get_distribution
import importlib

__version__ = get_distribution("perceval-quandela").version

from .components import *
from .backends import *
from .utils import *
from .rendering import *
from .runtime import *


def register_plugin(name, silent=False):
    try:
        plugin = importlib.import_module(name)
        assert plugin.register(silent) is True
    except Exception as e:
        raise RuntimeError("cannot import %s: %s" % (name, str(e)))
    return True
