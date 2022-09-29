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

from .runner import Runner
from perceval.platforms.job import Job
from perceval.platforms import Platform, RemoteJob, LocalJob
from perceval.serialization import deserialize_state, deserialize_state_list, deserialize_float


class Sampler(Runner):
    def __init__(self, platform: Platform, cu):
        super().__init__(platform)
        self.circuit = cu

    @property
    def sample(self) -> Job:
        if self._platform.is_remote():
            return RemoteJob(self._backend.async_sample, self._platform, deserialize_state)
        else:
            return LocalJob(self._backend.sample)

    @property
    def samples(self) -> Job:
        if self._platform.is_remote():
            return RemoteJob(self._backend.async_samples, self._platform, deserialize_state_list)
        else:
            return LocalJob(self._backend.samples)

    @property
    def sample_count(self) -> Job:
        if self._platform.is_remote():
            return RemoteJob(self._backend.async_sample_count, self._platform, lambda i:i)
        else:
            return LocalJob(self._backend.samples)

    @property
    def prob(self) -> Job:
        if self._platform.is_remote():
            return RemoteJob(self._backend.async_prob, self._platform, deserialize_float)
        else:
            return LocalJob(self._backend.prob)
