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

from perceval.runtime import RemoteJob, Job
from .abstract_algorithm import AAlgorithm
from perceval.components.abstract_processor import AProcessor


class QRNG(AAlgorithm):
    r"""
    Parameters:
        - `count`: int
        - `safest`: bool
    Return:
        - list[0, 1]
    """

    def __init__(self, processor: AProcessor):
        super().__init__(processor)

    @property
    def qrng(self) -> Job:
        assert self._processor.is_remote and "qrng" in self._processor.specs['available_commands'], \
            "Qrng algorithm is not implemented"
        return RemoteJob(lambda **kwargs: self._processor.async_execute("qrng", **kwargs),
                         self._processor.get_rpc_handler())

