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
from perceval.backends.cliffords2017 import CliffordClifford2017Backend
from perceval.backends.naive import NaiveBackend
from perceval.backends.slos import SLOSBackend
from perceval.backends.stepper import StepperBackend
from perceval.backends.strawberryfields import SFBackend

BACKEND_LIST = {
    CliffordClifford2017Backend.name.lower(): CliffordClifford2017Backend,
    NaiveBackend.name.lower(): NaiveBackend,
    SLOSBackend.name.lower(): SLOSBackend,
    StepperBackend.name.lower(): StepperBackend,
}
if SFBackend.is_available():
    BACKEND_LIST[SFBackend.name.lower()] = SFBackend

@deprecated(reason='Please use get_platform("platform name") instead')
class BackendFactory:
    def get_backend(self, backend_name="slos"):
        name = backend_name.lower()
        if name in BACKEND_LIST:
            return BACKEND_LIST[name]
        return BACKEND_LIST['slos']
