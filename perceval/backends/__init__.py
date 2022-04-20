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

from typing import Type, Union

from .template import Backend
from .cliffords2017 import CliffordClifford2017Backend
from .naive import NaiveBackend
from .slos import SLOSBackend
from .stepper import StepperBackend
from .strawberryfields import SFBackend


class BackendFactory:
    _backends = (NaiveBackend, CliffordClifford2017Backend, SFBackend, SLOSBackend, StepperBackend)

    def get_backend(self,
                    name: Union[str, None] = None) \
            -> Type[Backend]:
        """Returns a simulator backend

        :param name: The name of the simulator
        :return: the backend
        """
        if name is None:
            name = "SLOS"
        for backend in self._backends:
            if backend.name == name:
                return backend
        # TODO: check this exception
        raise ValueError("Unknown backend: %s" % name)

    def list_backend(self):
        return [backend.name for backend in self._backends]
