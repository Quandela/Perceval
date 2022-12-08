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

from multipledispatch import dispatch

from perceval.components.abstract_processor import AProcessor, ProcessorType
from perceval.components.linear_circuit import Circuit, ACircuit
from perceval.components.source import Source
from perceval.components.port import PortLocation, APort, LogicalState
from perceval.utils import BasicState
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
    def __init__(self, name: str, token: str, url: str = QUANDELA_CLOUD_URL, m: int = None):
        super().__init__()
        self.name = name

        self._rpc_handler = RPCHandler(self.name, url, token)
        self._specs = {}
        self._type = ProcessorType.SIMULATOR
        self._backend: RemoteBackend = None
        self._available_circuit_parameters = {}
        self.fetch_data()
        if m is not None:
            self._n_moi = m

        self._thresholded_output = "detector" in self._specs and self._specs["detector"] == "threshold"

    @property
    def is_remote(self) -> bool:
        return True

    def thresholded_output(self, value: bool):
        r"""
        Simulate threshold detectors on output states. All detections of more than one photon on any given mode is
        changed to 1. Some QPU and simulators can only perform threshold detection.

        :param value: enables threshold detection when True, otherwise disables it.
        """
        if value is False:
            assert not ("detector" in self._specs and self._specs["detector"] == "threshold"), \
                "given processor can only perform threshold detection"
        self.set_parameter("thresholded", value)
        super().thresholded_output(value)

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
        super().set_circuit(circuit)
        return self

    def add_port(self, m, port: APort, location: PortLocation = PortLocation.IN_OUT):
        # TODO: Remove this
        raise NotImplementedError("Ports not implemented for now with RemoteProcessors")

    def add_herald(self, mode: int, expected: int, name: str = None):
        # TODO: Remove this
        raise NotImplementedError("Heralds not implemented for now with RemoteProcessors")

    def __build_backend(self):
        # TODO: allow no circuit
        if self._n_moi is None:
            raise RuntimeError("No circuit set in RemoteProcessor")
        self._backend = RemoteBackend(self._rpc_handler, self.linear_circuit())

    def get_rpc_handler(self):
        return self._rpc_handler

    @property
    def type(self) -> ProcessorType:
        return self._type

    @dispatch(LogicalState)
    def with_input(self, input_state: LogicalState) -> None:
        r"""
        Set up the processor input with a LogicalState. Computes the input probability distribution.

        :param input_state: A LogicalState of length the input port count. Enclosed values have to match with ports
        encoding.
        """
        self._with_logical_input(input_state)

    @dispatch(BasicState)
    def with_input(self, input_state: BasicState) -> None:
        if 'max_photon_count' in self.constraints and input_state.n > self.constraints['max_photon_count']:
            raise RuntimeError(
                f"Too many photons in input state ({input_state.n} > {self.constraints['max_photon_count']})")
        if 'min_photon_count' in self.constraints and input_state.n < self.constraints['min_photon_count']:
            raise RuntimeError(
                f"Not enough photons in input state ({input_state.n} < {self.constraints['min_photon_count']})")
        if self._n_moi is not None and input_state.m != self._n_moi:
            raise RuntimeError(f"Input state and circuit size do not match ({input_state.m} != {self._n_moi})")
        self._input_state = input_state

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
        if self._n_moi is None:
            return 0
        return self._n_moi

    def add(self, mode_mapping, component, keep_port=True):
        if not isinstance(component, ACircuit):
            raise NotImplementedError("Non linear components not implemented for RemoteProcessors")
        self._backend = None
        super().add(mode_mapping, component, keep_port)

    def _compose_processor(self, connector, processor, keep_port: bool):
        assert isinstance(processor, RemoteProcessor), "can not mix types of processors"
        assert self.name == processor.name, "can not compose processors with different targets"
        assert self._rpc_handler.token == processor.get_rpc_handler().token, "can not compose processors with different tokens"
        assert self._rpc_handler.url == processor.get_rpc_handler().url, "can not compose processors with different url"
        super()._compose_processor(connector, processor, keep_port)

    @property
    def source(self):
        return None

    @source.setter
    def source(self, source: Source):
        # TODO: Implement source setter, setting parameters to be sent remotely
        raise NotImplementedError("Source setting not implemented for remote processors")
