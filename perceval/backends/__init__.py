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
import warnings

from ._abstract_backends import ABackend, ASamplingBackend, AProbAmpliBackend
from ._clifford2017 import Clifford2017Backend
from ._naive import NaiveBackend
from ._slos import SLOSBackend
from ._mps import MPSBackend


BACKEND_LIST = {
    "CliffordClifford2017": Clifford2017Backend,
    "MPS": MPSBackend,
    "Naive": NaiveBackend,
    "SLOS": SLOSBackend
}


class BackendFactory:
    @staticmethod
    def get_backend(backend_name: str = "SLOS") -> ABackend:
        name = backend_name
        if name in BACKEND_LIST:
            return BACKEND_LIST[name]()
        warnings.warn(f'Backend "{name}" not found. Falling back on SLOS')
        return BACKEND_LIST['SLOS']()

    @staticmethod
    def list():
        return list(BACKEND_LIST.keys())
