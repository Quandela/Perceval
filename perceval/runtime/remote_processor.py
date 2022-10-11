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

from typing import Dict, List, Callable

from perceval.components.abstract_processor import AProcessor, ProcessorType
from perceval.components import Circuit
from perceval.utils import Parameter, BasicState, SVDistribution, generate_sync_methods
from .remote_backend import RemoteBackend
from .remote_job import RemoteJob
from .rpc_handler import RPCHandler

QUANDELA_CLOUD_URL = 'https://api.cloud.quandela.dev'


def _extract_commands(specs):
    for v in specs.values():
        if 'available_commands' in v:
            for c in v['available_commands']:
                yield c


def _get_first_spec(specs, name):
    for v in specs.values():
        if name in v:
            return v[name]
    return None


def _split_platform_and_backend_name(name: str):
    backend_name = 'Naive'
    platform_name = name
    names = name.split(":")
    if len(names) == 2:
        platform_name = names[0]
        backend_name = names[1]
    return platform_name, backend_name


@generate_sync_methods
class RemoteProcessor(AProcessor):
    def __init__(self, name: str, token: str, url: str = QUANDELA_CLOUD_URL):
        super().__init__()
        (self.name, self._backend_name) = _split_platform_and_backend_name(name)

        self._rpc_handler = RPCHandler(self.name, url, token)
        self._specs = {}
        self._type = ProcessorType.SIMULATOR
        self._circuit = None
        self._backend: RemoteBackend = None
        self._available_circuit_parameters = {}
        self._input_state = None
        self.fetch_data()

    @property
    def is_remote(self):
        return True

    def fetch_data(self):
        platform_details = self._rpc_handler.fetch_platform_details()
        plugins_specs = platform_details['specs']
        self._specs.update(platform_details['specs'][next(iter(plugins_specs.keys()))])
        if platform_details['type'] != 'simulator':
            self._type = ProcessorType.PHYSICAL

    @property
    def specs(self):
        return self._specs

    def set_circuit(self, circuit: Circuit):
        self._circuit = circuit
        self.__build_backend()

    def __build_backend(self):
        if self._circuit is None:
            raise RuntimeError("No circuit set in RemoteProcessor")

        self._backend = RemoteBackend(self._rpc_handler, self._backend_name, self._circuit)

    def get_rpc_handler(self):
        return self._rpc_handler

    @property
    def type(self) -> ProcessorType:
        return self._type

    def with_input(self, input_state: BasicState) -> None:
        self._input_state = input_state

    @property
    def available_sampling_method(self) -> str:
        for k, v in _extract_commands(self._specs):
            return v
        return None

    def async_samples(self, count):
        if self._backend is None:
            self.__build_backend()

        return self._backend.async_samples(self._input_state, count, parameters=self._parameters)

    def async_sample_count(self, count) -> str:
        if self._backend is None:
            self.__build_backend()

        return self._backend.async_sample_count(self._input_state, count, parameters=self._parameters)

    def async_probs(self) -> SVDistribution:
        if self._backend is None:
            self.__build_backend()

        return self._backend.async_probs(self._input_state, parameters=self._parameters)

    def async_execute(self, command: str, **args):
        if self._backend is None:
            self.__build_backend()
        return self._backend.async_execute(command, parameters=self._parameters, **args)

    def get_circuit_parameters(self) -> Dict[str, Parameter]:
        pass

    def set_circuit_parameters(self, params: Dict[str, Parameter]) -> None:
        pass

    def resume_job(self, job_id:str, deserializer: Callable = None):
        job = RemoteJob(rpc_handler=self._rpc_handler, deserializer=deserializer)
        job.status()
        return job
