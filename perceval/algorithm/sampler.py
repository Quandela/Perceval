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
from typing import Callable, Dict, List

from .abstract_algorithm import AAlgorithm
from perceval.utils import BasicState, SVDistribution
from perceval.components.abstract_processor import AProcessor
from perceval.platforms.job import Job
from perceval.platforms import RemoteJob, LocalJob
from perceval.serialization import deserialize_state, deserialize_state_list, deserialize_float,\
    deserialize_sample_count


# Conversion functions (samples <=> probs <=> sample_count)
def samples_to_sample_count(sample_list: List[BasicState]) -> Dict[BasicState, int]:
    results = {}
    for s in sample_list:
        if s not in results:
            results[s] = sample_list.count(s)
    return results


def samples_to_probs(sample_list: List[BasicState]) -> SVDistribution:
    return sample_count_to_probs(samples_to_sample_count(sample_list))


def probs_to_sample_count(probs: SVDistribution, count: int) -> Dict[BasicState, int]:
    return samples_to_sample_count(probs.sample(count))


def probs_to_samples(probs: SVDistribution, count: int) -> List[BasicState]:
    return probs.sample(count)


def sample_count_to_probs(sample_count: Dict[BasicState, int]):
    svd = SVDistribution()
    n_samples = 0
    for state, count in sample_count.items():
        if count == 0:
            continue
        if count < 0:
            raise RuntimeError(f"A sample count must be positive (got {count})")
        svd[state] = count
        n_samples += count
    for state, value in svd.items():
        svd[state] = value / n_samples
    return svd


def sample_count_to_samples(sample_count: Dict[BasicState, int], count: int):
    return sample_count_to_probs(sample_count).sample(count)


class Sampler(AAlgorithm):
    PROBS_SIMU_SAMPLE_COUNT = 10000  # Arbitrary value

    def __init__(self, processor: AProcessor):
        super().__init__(processor)
        self._sample_count_mapping = {
            'probs': self._sample_count_from_probs,
            'sample_count': self._processor.sample_count,
            'samples': self._sample_count_from_samples
        }
        self._samples_mapping = {
            'probs': self._samples_from_probs,
            'sample_count': self._samples_from_sample_count,
            'samples': self._processor.samples
        }
        self._probs_mapping = {
            'probs': self._processor.probs,
            'sample_count': self._probs_from_sample_count,
            'samples': self._probs_from_samples
        }

    def _sample_count_from_samples(self, count: int, progress_callback: Callable = None):  # signature of sample_count()
        sample_list = self._processor.samples(count, progress_callback)
        return samples_to_sample_count(sample_list)

    def _sample_count_from_probs(self, count: int, progress_callback: Callable = None):
        probs = self._processor.probs(progress_callback)
        return probs_to_sample_count(probs, count)

    def _probs_from_samples(self, progress_callback: Callable = None):
        count = self.PROBS_SIMU_SAMPLE_COUNT
        sample_list = self._processor.samples(count, progress_callback)
        return samples_to_probs(sample_list)

    def _probs_from_sample_count(self, progress_callback: Callable = None):
        count = self.PROBS_SIMU_SAMPLE_COUNT
        sample_count = self._processor.sample_count(count, progress_callback)
        return sample_count_to_probs(sample_count)

    def _samples_from_sample_count(self, count: int, progress_callback: Callable = None):
        sample_count = self._processor.sample_count(count, progress_callback)
        return sample_count_to_samples(sample_count, count)

    def _samples_from_probs(self, count: int, progress_callback: Callable = None):
        probs = self._processor.probs(progress_callback)
        return probs_to_samples(probs, count)

    @property
    def samples(self) -> Job:
        if self._processor.is_remote:
            return RemoteJob(self._backend.async_samples, self._platform, deserialize_state_list)
        else:
            try:
                method = self._samples_mapping[self._processor.available_sampling_method]
            except KeyError:
                raise NotImplementedError(
                    f"Method to retrieve samples from {self._processor.available_sampling_method} not implemented")
            return LocalJob(method)

    @property
    def sample_count(self) -> Job:
        if self._processor.is_remote:
            return RemoteJob(self._backend.async_sample_count, self._platform, deserialize_sample_count)
        else:
            try:
                method = self._sample_count_mapping[self._processor.available_sampling_method]
            except KeyError:
                raise NotImplementedError(
                    f"Method to retrieve sample_count from {self._processor.available_sampling_method} not implemented")
            return LocalJob(method)

    @property
    def probs(self) -> Job:
        if self._processor.is_remote:
            return RemoteJob(self._backend.async_prob, self._platform, deserialize_float)
        else:
            try:
                method = self._probs_mapping[self._processor.available_sampling_method]
            except KeyError:
                raise NotImplementedError(
                    f"Method to retrieve probs from {self._processor.available_sampling_method} not implemented")
            return LocalJob(method)
