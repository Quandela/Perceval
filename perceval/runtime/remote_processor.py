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

from typing import Dict, List

from perceval.components.abstract_processor import AProcessor, ProcessorType
from perceval.components.linear_circuit import Circuit
from perceval.components.processor import Processor
from perceval.utils import BasicState, Parameter
from perceval.serialization import deserialize
from .remote_backend import RemoteBackend
from .remote_job import RemoteJob
from .rpc_handler import RPCHandler

QUANDELA_CLOUD_URL = 'https://api.cloud.quandela.com'


def _get_first_spec(specs, name):
    for v in specs.values():
        if name in v:
            return v[name]
    return None


class RemoteProcessor(AProcessor):
    def __init__(self, name: str, token: str, url: str = QUANDELA_CLOUD_URL):
        super().__init__()
        self.name = name

        self._rpc_handler = RPCHandler(self.name, url, token)
        self._specs = {}
        self._type = ProcessorType.SIMULATOR
        self._circuit = None
        self._backend: RemoteBackend = None
        self._available_circuit_parameters = {}
        self._input_state = None
        self.fetch_data()
        # Just to avoid rewriting of some functions that should not belong to the AbstractProcessor
        self._local_processor: Processor = None

    @property
    def is_remote(self) -> bool:
        return True

    @property
    def is_threshold(self) -> bool:  # Supposes that unspecified means no threshold
        return "detector" in self._specs and self._specs["detector"] == "threshold"

    def fetch_data(self):
        platform_details = self._rpc_handler.fetch_platform_details()
        platform_specs = deserialize(platform_details['specs'])
        self._specs.update(platform_specs)
        if platform_details['type'] != 'simulator':
            self._type = ProcessorType.PHYSICAL

    @property
    def specs(self):
        return self._specs

    @property
    def constraints(self) -> Dict:
        if 'constraints' in self._specs:
            return self._specs['constraints']
        return {}

    def set_circuit(self, circuit: Circuit):
        if 'max_mode_count' in self.constraints and circuit.m > self.constraints['max_mode_count']:
            raise RuntimeError(f"Circuit too big ({circuit.m} modes > {self.constraints['max_mode_count']})")
        if 'min_mode_count' in self.constraints and circuit.m < self.constraints['min_mode_count']:
            raise RuntimeError(f"Circuit too small ({circuit.m} < {self.constraints['min_mode_count']})")
        if self._input_state is not None and self._input_state.m != circuit.m:
            raise RuntimeError(f"Circuit and input state size do not match ({circuit.m} != {self._input_state.m})")
        self._circuit = circuit
        self.__build_backend()
        self._create_local_processor()

    def __build_backend(self):
        if self._circuit is None:
            raise RuntimeError("No circuit set in RemoteProcessor")

        self._backend = RemoteBackend(self._rpc_handler, self._circuit)

    def get_rpc_handler(self):
        return self._rpc_handler

    @property
    def type(self) -> ProcessorType:
        return self._type

    def with_input(self, input_state: BasicState) -> None:
        if 'max_photon_count' in self.constraints and input_state.n > self.constraints['max_photon_count']:
            raise RuntimeError(
                f"Too many photons in input state ({input_state.n} > {self.constraints['max_photon_count']})")
        if 'min_photon_count' in self.constraints and input_state.n < self.constraints['min_photon_count']:
            raise RuntimeError(
                f"Not enough photons in input state ({input_state.n} < {self.constraints['min_photon_count']})")
        if self._circuit is not None and input_state.m != self._circuit.m:
            raise RuntimeError(f"Input state and circuit size do not match ({input_state.m} != {self._circuit.m})")
        self._input_state = input_state
        if self._local_processor is not None:
            self._local_processor.with_input(input_state)
            self._local_processor.mode_post_selection(self.parameters["mode_post_select"]
                                                      if "mode_post_select" in self.parameters
                                                      else input_state.n)

    def mode_post_selection(self, n: int):
        super().mode_post_selection(n)
        if self._local_processor is not None:
            self._local_processor.mode_post_selection(n)

    @property
    def available_commands(self) -> List[str]:
        return self._specs.get("available_commands", [])

    def async_samples(self, count, **args) -> str:
        if self._backend is None:
            self.__build_backend()
        return self._backend.async_execute("samples", self._parameters, input_state=self._input_state, count=count, **args)

    def async_sample_count(self, count, **args) -> str:
        if self._backend is None:
            self.__build_backend()
        return self._backend.async_execute("sample_count", self._parameters, input_state=self._input_state, count=count, **args)

    def async_probs(self, **args) -> str:
        if self._backend is None:
            self.__build_backend()
        return self._backend.async_execute("probs", self._parameters, input_state=self._input_state, **args)

    def async_execute(self, command: str, **args) -> str:
        if self._backend is None:
            self.__build_backend()
        return self._backend.async_execute(command, parameters=self._parameters, **args)

    def resume_job(self, job_id: str) -> RemoteJob:
        return RemoteJob.from_id(job_id, self._rpc_handler)

    @property
    def m(self) -> int:
        if self._circuit is None:
            return 0
        return self._circuit.m

    @property
    def post_select_fn(self):
        return self._local_processor.post_select_fn

    def _create_local_processor(self):
        proc = Processor("SLOS", self._circuit)
        if self._input_state is not None:
            proc.with_input(self._input_state)
            proc.mode_post_selection(self.parameters["mode_post_select"] if "mode_post_select" in self.parameters
                                     else self._input_state.n)
        proc.thresholded_output(self.is_threshold)
        self._local_processor = proc

    def set_postprocess(self, postprocess_func):
        r"""
        Set or remove a logical post-selection function. Unused for now.

        :param postprocess_func: Sets a post-selection function. Its signature must be `func(s: BasicState) -> bool`.
            If None is passed as parameter, removes the previously defined post-selection function.
        """
        self._local_processor.set_postprocess(postprocess_func)

    def postprocess_output(self, s: BasicState) -> BasicState:
        # Apply threshold if exists
        return self._local_processor.postprocess_output(s)

    def _state_selected_physical(self, output_state: BasicState) -> bool:
        return self._local_processor._state_selected_physical(output_state)

    def get_circuit_parameters(self) -> Dict[str, Parameter]:
        return self._local_processor.get_circuit_parameters()
