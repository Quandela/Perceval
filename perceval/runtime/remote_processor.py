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
from requests import HTTPError
from copy import copy

from perceval.components import ACircuit, AComponent, Experiment, PortLocation
from perceval.utils import FockState, NoiseModel, PostSelect, ProcessorType
from perceval.utils.logging import get_logger, channel
from perceval.serialization import deserialize

from .platform_specs import PlatformSpecs
from .remote_job import RemoteJob
from .rpc_handler import RPCHandler
from .remote_config import RemoteConfig
from .payload_generator import PayloadGenerator
from .abstract_processor import AProcessor
from .processor import Processor

PERFS_KEY = "perfs"
TRANSMITTANCE_KEY = "Transmittance (%)"
INDISTINGUISHABILITY_KEY = "HOM (%)"
G2_KEY = "g2 (%)"
DEFAULT_TRANSMITTANCE = 0.06


def perf_dict_to_noise(perfs: dict[str, float]) -> NoiseModel:
    nm = NoiseModel()
    if TRANSMITTANCE_KEY in perfs:
        nm.transmittance = perfs[TRANSMITTANCE_KEY] / 100
    if INDISTINGUISHABILITY_KEY in perfs:
        nm.indistinguishability = perfs[INDISTINGUISHABILITY_KEY] / 100
    if G2_KEY in perfs:
        nm.g2 = perfs[G2_KEY] / 100
    return nm


class RemoteProcessor(AProcessor):
    @staticmethod
    def from_local_processor(
            processor: Processor,
            name: str = None,
            token: str = None,
            url: str = None,
            proxies: dict[str,str] = None,
            rpc_handler: RPCHandler = None):
        rp = RemoteProcessor(
            name=name,
            token=token,
            url=url,
            proxies=proxies,
            rpc_handler=rpc_handler)
        rp.noise = processor.noise
        rp.add(0, processor)
        rp.min_detected_photons_filter(processor._min_detected_photons_filter)
        if processor.input_state is not None:
            rp.with_input(processor.input_state)
        return rp

    def __init__(self,
                 name: str = None,
                 token: str = None,
                 url: str = None,
                 proxies: dict[str,str] = None,
                 rpc_handler: RPCHandler = None,
                 m: int = None,
                 noise: NoiseModel = None):
        """
        :param name: Platform name
        :param token: Token value to authenticate the user
        :param url: Base URL for the Cloud API to connect to
        :param proxies: Dictionary mapping protocol to the URL of the proxy
        :param rpc_handler: Inject an already constructed Remote Procedure Call handler (alternative init);
            when doing so, name, token and url are expected to be blank
        :param m: Initialize the processor to a given size (number of modes). If not set here, the first component or
            circuit added decides of the processor size
        :param noise: a NoiseModel containing noise parameters (defaults to no noise)
                      simulated noise is ignored when working on a physical Quantum Processing Unit
        """
        super().__init__(Experiment(m, noise=noise, name=name))
        if rpc_handler is not None:  # When a rpc_handler object is passed, name, token and url are expected to be None
            self._rpc_handler = rpc_handler
            self.name = rpc_handler.name  # Here, we are mixing the experiment name and the Processor name
            if name is not None and name != self.name:
                get_logger().warn(
                    f"Initialised a RemoteProcessor with two different platform names ({self.name} vs {name})", channel.user)
            self.proxies = rpc_handler.proxies
        else:
            remote = RemoteConfig()
            if name is None:
                raise ValueError("Parameter 'name' must have a value")
            if token is None:
                token = remote.get_token()
            if not token:
                raise ConnectionError("No token found")
            if url is None:
                url = remote.get_url()
            if proxies is None:
                proxies = remote.get_proxies()
            self.name = name
            self.proxies = proxies
            self._rpc_handler = RPCHandler(self.name, url, token, proxies)

        self._specs = PlatformSpecs()
        self._perfs = {}
        self._status = None
        self._available_circuit_parameters = {}
        self.fetch_data()
        get_logger().info(f"Connected to Cloud platform {self.name}", channel.general)

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name: str):
        self._name = name
        self._experiment.name = name

    def _circuit_change_observer(self, new_component: Experiment | AComponent = None):
        pass
        # TODO: Check that the component matches what the platform can do
        # if new_component is not None:
        #     if isinstance(new_component, Experiment):
        #         if not new_component.is_unitary:
        #             raise RuntimeError('Cannot compose a RemoteProcessor with a processor containing non linear components')
        #         if new_component.has_feedforward:
        #             raise RuntimeError('Cannot compose a RemoteProcessor with a processor containing feed-forward')
        #
        #     elif not isinstance(new_component, IDetector) and not isinstance(new_component, ACircuit):
        #         raise NotImplementedError("Non linear components not implemented for RemoteProcessors")

    def _noise_changed_observer(self):
        if self.noise and self.type == ProcessorType.PHYSICAL:  # Injecting a noise model to an actual QPU makes no sense
            get_logger().warn(
                f"{self.name} is not a simulator but an actual QPU: user defined noise parameters will be ignored",
                channel.user)

    @property
    def is_remote(self) -> bool:
        return True

    def fetch_data(self):
        try:
            platform_details = self._rpc_handler.fetch_platform_details()
        except HTTPError as e:
            raise HTTPError(f"Error while fetching platform details: {e}") from None
        self._status = platform_details.get("status")
        platform_specs = deserialize(platform_details['specs'], strict=False)
        self._specs.update(platform_specs)  # No verification here, we suppose every check was made by the platform
        self._specs["type"] = platform_details.get('type', "simulator")
        if PERFS_KEY in platform_details:
            self._perfs.update(platform_details[PERFS_KEY])

    @property
    def specs(self) -> PlatformSpecs:
        return self._specs

    @property
    def performance(self) -> dict:
        return self._perfs

    def get_current_noise(self) -> NoiseModel:
        """
        :return: The NoiseModel associated to the last performance characterization of the remote platform.
            Simulators without hardware return a NoiseModel with a default transmittance value.
            Note that, depending on the platform, the performance dictionary may have more information that can't be translated into NoiseModel
        """
        perfs = copy(self._perfs)
        if TRANSMITTANCE_KEY not in self._perfs:
            perfs[TRANSMITTANCE_KEY] = DEFAULT_TRANSMITTANCE * 100
            get_logger().warn(f"No transmittance was found for {self.name}, using default {DEFAULT_TRANSMITTANCE}", channel.user)
        return perf_dict_to_noise(perfs)

    @property
    def constraints(self) -> dict:
        return self._specs.constraints

    @property
    def status(self) -> str:
        return self._status

    def check_circuit_size(self, m: int):
        if 'max_mode_count' in self.constraints and m > self.constraints['max_mode_count']:
            raise RuntimeError(f"Circuit too big ({m} modes > {self.constraints['max_mode_count']})")
        if 'min_mode_count' in self.constraints and m < self.constraints['min_mode_count']:
            raise RuntimeError(f"Circuit too small ({m} < {self.constraints['min_mode_count']})")
        if self.input_state is not None and self.input_state.m != m:
            raise RuntimeError(f"Circuit and input state size do not match ({m} != {self.input_state.m})")

    def check_circuit(self, circuit: ACircuit):
        self.check_circuit_size(circuit.m)

    def set_circuit(self, circuit: ACircuit):
        self.check_circuit(circuit)
        super().set_circuit(circuit)
        return self

    def get_rpc_handler(self):
        return self._rpc_handler

    @property
    def type(self) -> ProcessorType:
        return self._specs.type

    def check_input(self, input_state: FockState) -> None:
        super().check_input(input_state)
        n_heralds = sum(self.in_heralds.values())
        n_photons = input_state.n + n_heralds
        if 'max_photon_count' in self.constraints and n_photons > self.constraints['max_photon_count']:
            raise RuntimeError(
                f"Too many photons in input state ({input_state.n} + {n_heralds} heralds > {self.constraints['max_photon_count']})")
        if 'min_photon_count' in self.constraints and n_photons < self.constraints['min_photon_count']:
            raise RuntimeError(
                f"Not enough photons in input state ({n_photons} < {self.constraints['min_photon_count']})")
        if ('support_multi_photon' in self.constraints and not self.constraints['support_multi_photon']
                and not all(mode_photon_cnt <= 1 for mode_photon_cnt in input_state)):
            raise RuntimeError(f"Input state ({input_state}) is not permitted. QPU/QPU simulators accept more than "
                               f"1 photon per mode:{self.constraints['support_multi_photon']})")
        if self.m_in is not None and input_state.m != self.m_in:
            raise RuntimeError(f"Input state and circuit size do not match ({input_state.m} != {self.m_in})")

    @property
    def available_commands(self) -> list[str]:
        return self._specs.available_commands

    def prepare_job_payload(self, command: str, **kwargs) -> dict[str, any]:
        self.check_min_detected_photons_filter()
        self.check_circuit_size(self.circuit_size)
        if self.input_state:
            self.check_input(self.remove_in_heralded_modes(self.input_state))

        payload = PayloadGenerator.generate_payload(command, self.experiment, self._parameters, self.name, **kwargs)

        self.log_resources(command, self._parameters)
        return payload

    def resume_job(self, job_id: str) -> RemoteJob:
        return RemoteJob.from_id(job_id, self._rpc_handler)

    def _compute_sample_of_interest_probability(self, param_values: dict = None) -> float:
        # Simulation with a noisy source (only losses)
        nm = self.get_current_noise()
        nm.g2 = 0
        nm.indistinguishability = 1

        n = self.input_state.n
        photon_filter = n
        if self._min_detected_photons_filter is not None:
            photon_filter = self._min_detected_photons_filter + sum(self.heralds.values())
            if photon_filter > n:
                return 0
        if photon_filter < 2:
            return 1

        exp = self.experiment.copy()
        if param_values is not None:
            params = exp.get_circuit_parameters()
            for param_name, value in param_values.items():
                if param_name in params:
                    params[param_name].set_value(value)
        lp = Processor("SLAP", exp, nm)

        # Remove all selection
        lp.min_detected_photons_filter(1)
        lp.set_postselection(PostSelect())
        while len(lp.in_heralds):
            m = next(iter(lp.in_heralds))
            lp.remove_port(m, PortLocation.INPUT)
        while len(lp.heralds):
            m = next(iter(lp.heralds))
            lp.remove_port(m, PortLocation.OUTPUT)

        archi = self.specs.architecture
        if archi is not None:
            for m in range(lp.circuit_size):
                lp.add(m, archi.detectors[m])

        lp.with_input(self.input_state)
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

    def log_resources(self, command: str, extra_parameters: dict):
        """Log resources of the remote processor

        :param command: name of the method used
        :param extra_parameters: extra parameters to log
        """
        extra_parameters = {key: value for key, value in extra_parameters.items() if value is not None}
        my_dict = {
            'layer': 'RemoteProcessor',
            'platform': self.name,
            'm': self.circuit_size,
            'command': command
        }
        if self.input_state:
            my_dict['n'] = self.input_state.n
        if self.noise:
            my_dict['noise'] = self.noise.__dict__()
        if extra_parameters:
            my_dict.update(extra_parameters)
        get_logger().log_resources(my_dict)

    def compute_physical_logical_perf(self, value: bool):
        """
        Split performance into the physical and logical parts - when available

        :param value: True to compute the physical and logical performances, False otherwise.
        """
        self.set_parameter("compute_physical_logical_perf", value)
