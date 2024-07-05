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

from perceval.components.abstract_processor import AProcessor


class AAlgorithm:
    _MAX_SHOTS_NAMED_PARAM = "max_shots_per_call"

    def __init__(self, processor: AProcessor, **kwargs):
        self._processor = processor
        if not self._check_compatibility():
            raise RuntimeError("Processor and algorithm are not compatible")

        self.default_job_name = None

        self._max_shots = kwargs.get(self._MAX_SHOTS_NAMED_PARAM)
        if self._max_shots:
            self._max_shots = int(self._max_shots)
            if self._max_shots < 1:
                raise RuntimeError(f'`{self._MAX_SHOTS_NAMED_PARAM}` must be a positive value')
        # max_shots_per_call must be found in **kwargs when the processor is remote.
        # This condition is forced because the user will consume credits on the cloud and needs to set an upper bound
        if processor.is_remote and not self._max_shots:
            raise RuntimeError(f'Please input a `{self._MAX_SHOTS_NAMED_PARAM}` value when using a RemoteProcessor')

    def _check_compatibility(self) -> bool:
        # if self._processor.is_remote:
        #     # TODO remote compatibility check
        #     return False
        return True
