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
import uuid
from typing import Dict, List
from multipledispatch import dispatch

from perceval.components.abstract_processor import AProcessor, ProcessorType
from perceval.components import ACircuit, Processor, Source
from perceval.components.port import PortLocation, APort
from perceval.utils import BasicState, LogicalState, PMetadata
from perceval.serialization import deserialize, serialize
from .remote_job import RemoteJob
from .rpc_handler import RPCHandler
from ._token_management import TokenProvider

__process_id__ = uuid.uuid4()

QUANDELA_CLOUD_URL = 'https://api.cloud.quandela.com'


class RemoteProcessor(AProcessor):
    def __init__(self, name: str, token: str = None, url: str = QUANDELA_CLOUD_URL, m: int = None):
        super().__init__()
        self.name = name
        if token is None:
            provider = TokenProvider()
            token = provider.get_token()
        if token is None:
            raise ConnectionError("No token found")

        self._rpc_handler = RPCHandler(self.name, url, token)
        self._specs = {}
        self._type = ProcessorType.SIMULATOR
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

    def set_circuit(self, circuit: ACircuit):
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
        self.check_input(input_state)
        self._input_state = input_state

    def check_input(self, input_state: BasicState) -> None:
        if 'max_photon_count' in self.constraints and input_state.n > self.constraints['max_photon_count']:
            raise RuntimeError(
                f"Too many photons in input state ({input_state.n} > {self.constraints['max_photon_count']})")
        if 'min_photon_count' in self.constraints and input_state.n < self.constraints['min_photon_count']:
            raise RuntimeError(
                f"Not enough photons in input state ({input_state.n} < {self.constraints['min_photon_count']})")
        if self._n_moi is not None and input_state.m != self._n_moi:
            raise RuntimeError(f"Input state and circuit size do not match ({input_state.m} != {self._n_moi})")

    @property
    def available_commands(self) -> List[str]:
        return self._specs.get("available_commands", [])

    def prepare_job_payload(self, command: str, circuitless: bool = False, inputless: bool = False, **kwargs):
        j = {
            'platform_name': self.name,
            'pcvl_version': PMetadata.short_version(),
            'process_id': str(__process_id__)
        }
        payload = {
            'command': command,
            **kwargs
        }
        if self._components and not circuitless:
            payload['circuit'] = serialize(self.linear_circuit())
        if self._input_state and not inputless:
            payload['input_state'] = serialize(self._input_state)
        if self._parameters:
            payload['parameters'] = self._parameters
        j['payload'] = payload
        return j

    def resume_job(self, job_id: str) -> RemoteJob:
        return RemoteJob.from_id(job_id, self._rpc_handler)

    @property
    def m(self) -> int:
        if self._n_moi is None:
            return 0
        return self._n_moi

    def _add_component(self, mode_mapping, component):
        if not isinstance(component, ACircuit):
            raise NotImplementedError("Non linear components not implemented for RemoteProcessors")
        super()._add_component(mode_mapping, component)

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

    def _compute_sample_of_interest_probability(self,
                                                transmittance: float = 0.06,
                                                param_values: dict = None
                                                ) -> float:
        # TODO retrieve transmittance from the cloud worker
        losses = 1 - transmittance
        n = self._input_state.n
        photon_filter = n
        if self._min_detected_photons is not None:
            photon_filter = self._min_detected_photons
            if photon_filter > n:
                return 0
        if photon_filter < 2:
            return 1

        # Simulation with a noisy source (only losses)
        c = self.linear_circuit(flatten=True).copy()
        if param_values:
            for n, v in param_values.items():
                c.param(n).set_value(v)
        lp = Processor("SLOS", c, Source(losses=losses))
        lp.min_detected_photons_filter(1)
        lp.thresholded_output(self._thresholded_output)
        lp.with_input(self._input_state)
        probs = lp.probs()
        p_above_filter_ns = 0
        for state, prob in probs['results'].items():
            if state.n >= photon_filter:
                p_above_filter_ns += prob
        return p_above_filter_ns

    def estimate_required_shots(self, nsamples: int, param_values: dict = None) -> int:
        """
        Compute an estimate number of required shots given the platform and the user request.
        The circuit, input state, minimum photon filter are taken into account.

        :param nsamples: Number of expected samples of interest
        :param param_values: Key/value pairs for variable parameters inside the circuit. All parameters need to be fixed
            for this computation to run.
        :return: Estimate of the number of shots the user needs to acquire enough samples of interest
        """
        p_interest = self._compute_sample_of_interest_probability(param_values=param_values)
        if p_interest == 0:
            return None
        return round(nsamples / p_interest)

    def estimate_expected_samples(self, nshots: int, param_values: dict = None) -> int:
        """
        Compute an estimate number of samples the user can expect given the platform and the user request.
        The circuit, input state, minimum photon filter are taken into account.

        :param nshots: Number of shots the user is willing to consume
        :param param_values: Key/value pairs for variable parameters inside the circuit. All parameters need to be fixed
            for this computation to run.
        :return: Estimate of the number of samples of interest the user can expect back
        """
        p_interest = self._compute_sample_of_interest_probability(param_values=param_values)
        return round(nshots * p_interest)
