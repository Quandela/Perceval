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
from typing import Callable, Dict

from .abstract_algorithm import AAlgorithm
from perceval.utils import samples_to_sample_count, samples_to_probs, sample_count_to_samples, sample_count_to_probs,\
    probs_to_samples, probs_to_sample_count
from perceval.components.abstract_processor import AProcessor
from perceval.runtime import Job, RemoteJob, LocalJob
from perceval.serialization import deserialize_state, deserialize_state_list, deserialize_svdistribution,\
    deserialize_sample_count


class Sampler(AAlgorithm):
    """
    Base algorithm able to retrieve some sampling results via 3 methods
    - samples(count) : returns a list of sampled states
    - sample_count(count) : return a table (output state) : (sampled count)
                            the sum of 'sampled counts' can slightly differ from requested 'count'
    - probs() : returns a probability distribution of output states

    The form of the output for all 3 sampling methods is a dictionary containing a 'results' key and several performance
    values.
    """
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

    def _sample_count_from_samples(self, count: int, progress_callback: Callable = None) -> Dict:
        output = self._processor.samples(count, progress_callback)
        output['results'] = samples_to_sample_count(output['results'])
        return output

    def _sample_count_from_probs(self, count: int, progress_callback: Callable = None) -> Dict:
        output = self._processor.probs(progress_callback)
        output['results'] = probs_to_sample_count(output['results'], count)
        return output

    def _probs_from_samples(self, progress_callback: Callable = None) -> Dict:
        count = self.PROBS_SIMU_SAMPLE_COUNT
        output = self._processor.samples(count, progress_callback)
        output['results'] = samples_to_probs(output['results'])
        return output

    def _probs_from_sample_count(self, progress_callback: Callable = None) -> Dict:
        count = self.PROBS_SIMU_SAMPLE_COUNT
        output = self._processor.sample_count(count, progress_callback)
        output['results'] = sample_count_to_probs(output['results'])
        return output

    def _samples_from_sample_count(self, count: int, progress_callback: Callable = None) -> Dict:
        output = self._processor.sample_count(count, progress_callback)
        output['results'] = sample_count_to_samples(output['results'], count)
        return output

    def _samples_from_probs(self, count: int, progress_callback: Callable = None) -> Dict:
        output = self._processor.probs(progress_callback)
        output['results'] = probs_to_samples(output['results'], count)
        return output

    @property
    def samples(self) -> Job:
        if self._processor.is_remote:
            return RemoteJob(self._processor.async_samples, self._processor.get_rpc_handler())
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
            return RemoteJob(self._processor.async_sample_count, self._processor.get_rpc_handler())
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
            return RemoteJob(self._processor.async_probs, self._processor.get_rpc_handler())
        else:
            try:
                method = self._probs_mapping[self._processor.available_sampling_method]
            except KeyError:
                raise NotImplementedError(
                    f"Method to retrieve probs from {self._processor.available_sampling_method} not implemented")
            return LocalJob(method)
