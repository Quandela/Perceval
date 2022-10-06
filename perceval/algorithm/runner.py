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

from perceval.platforms.platform import Platform
from perceval.platforms.local_job import LocalJob
from perceval.platforms.remote_job import RemoteJob
from perceval.components import ACircuit
from perceval.utils import Matrix


class Runner:
    def __init__(self, platform: Platform):
        self._platform = platform
        self._cu = None
        self._backend = None
        self._job_type = RemoteJob if platform.is_remote() else LocalJob

    @property
    def circuit(self):
        return self._cu

    @circuit.setter
    def circuit(self, cu):
        assert isinstance(cu, ACircuit) or isinstance(cu, Matrix), \
            f'Runner accepts linear circuits or unitary matrix as input, not {type(cu)}'
        self._cu = cu
        self._backend = self._platform.backend(cu)
        if not self._check_compatibility():
            raise RuntimeError('Incompatible platform and circuit')

    def _check_compatibility(self) -> bool:
        # if self._platform.is_remote():
        #     # TODO remote compatibility check
        #     return False
        return True
