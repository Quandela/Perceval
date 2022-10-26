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

from deprecated import deprecated
from .template import Backend
from .cliffords2017 import CliffordClifford2017Backend
from .naive import NaiveBackend
from .slos import SLOSBackend
from .stepper import StepperBackend
from .strawberryfields import SFBackend
from .mps import MPSBackend


BACKEND_LIST = {
    CliffordClifford2017Backend.name: CliffordClifford2017Backend,
    MPSBackend.name: MPSBackend,
    NaiveBackend.name: NaiveBackend,
    SLOSBackend.name: SLOSBackend,
    StepperBackend.name: StepperBackend,
}
if SFBackend.is_available():
    BACKEND_LIST[SFBackend.name] = SFBackend

@deprecated(reason='Please use get_platform("platform name") instead')
class BackendFactory:
    def get_backend(self, backend_name="SLOS"):
        name = backend_name
        if name in BACKEND_LIST:
            return BACKEND_LIST[name]
        return BACKEND_LIST['SLOS']
