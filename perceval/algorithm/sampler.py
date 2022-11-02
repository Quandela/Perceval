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
from perceval.utils import samples_to_sample_count, samples_to_probs, sample_count_to_samples,\
                           sample_count_to_probs, probs_to_samples, probs_to_sample_count
from perceval.components.abstract_processor import AProcessor
from perceval.runtime import Job, RemoteJob, LocalJob


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
        self._method_mapping = {
            'probs': { 'sample_count': sample_count_to_probs, 'samples': samples_to_probs},
            'sample_count': { 'probs': probs_to_sample_count, 'samples': samples_to_sample_count},
            'samples': {'probs': probs_to_samples, 'sample_count': sample_count_to_samples}
        }

    def _get_primitive_converter(self, method: str):
        available_primitives = self._processor.available_commands
        if method in available_primitives:
            return method, None
        if method in self._method_mapping:
            pmap = self._method_mapping[method]
            for k, converter in pmap.items():
                if k in available_primitives:
                    return k, converter
        return None, None

    def _generic(self, method: str):
        primitive, converter = self._get_primitive_converter(method)
        delta_parameters = {}
        # adapt the parameters list
        if method.find('sample') != -1 and primitive.find('sample') == -1:
            delta_parameters['count'] = None
        elif method.find('sample') == -1 and primitive.find('sample') != -1:
            delta_parameters['count'] = self.PROBS_SIMU_SAMPLE_COUNT
        if primitive is None:
            raise NotImplementedError(f"cannot find primitive to execute {method} in {self._processor.available_commands}")
        if self._processor.is_remote:
            job_context = None
            if converter:
                job_context = {"result_mapping": ['perceval.utils', converter.__name__]}
            rj = RemoteJob(getattr(self._processor, "async_"+primitive),
                           self._processor.get_rpc_handler(), delta_parameters=delta_parameters,
                           job_context=job_context)
            return rj
        else:
            return LocalJob(getattr(self._processor, primitive),
                            result_mapping_function=converter,
                            delta_parameters=delta_parameters)


    @property
    def samples(self) -> Job:
        return self._generic("samples")

    @property
    def sample_count(self) -> Job:
        return self._generic("sample_count")

    @property
    def probs(self) -> Job:
        return self._generic("probs")
