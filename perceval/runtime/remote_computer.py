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

import time
from abc import ABC, abstractmethod
from copy import copy, deepcopy

from .computation import Computation
from .abstract_computer import AbstractComputer, AsyncGetter
from .computation_iterator import ComputationIterator
from .platform_specs import PlatformSpecs
from .error_mitigation import AbstractMitigation
from .job_status import RunningStatus
from .simulated_computer import SimulatedComputer
from .command import CommandFactory

from perceval.utils import perf_dict_to_noise, ProgressCallback, ProcessorType, NoiseModel, PostSelect
from perceval.utils.logging import channel, get_logger
from perceval.components import PortLocation


class CommunicationLayer(ABC):
    """
    This class is responsible for the communication with the distant computer.
    """

    @abstractmethod
    def get_specs(self) -> PlatformSpecs:
        """
        :return: The specs of the target platform
        """
        pass

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """
        :return: True if jobs can be sent to the target platform (queue is not full), False otherwise
        """
        pass

    @abstractmethod
    def send(self, payload: dict) -> AsyncGetter:
        pass

    @abstractmethod
    def get_results(self, async_getter: AsyncGetter) -> dict:
        pass

    @abstractmethod
    def get_status(self, async_getter: AsyncGetter) -> RunningStatus:
        pass

    @abstractmethod
    def get_performances(self) -> dict:
        pass

    @abstractmethod
    def get_commands(self) -> list[str]:
        pass

    @abstractmethod
    def cancel(self, async_getter: AsyncGetter) -> None:
        pass


class RemoteComputer(AbstractComputer):

    def __init__(self, communication_layer: CommunicationLayer):
        super().__init__()
        self._communication_layer = communication_layer  # cloud_access is the communication layer
        self._commands = communication_layer.get_commands()
        self._specs = communication_layer.get_specs()
        self._perfs = communication_layer.get_performances()
        self._custom_noise: NoiseModel | None = None
        self._remote_mitigations: list[AbstractMitigation] = []  # Mitigation setters must fill this by default
        # TODO: how to get default mitigations ?

    @property
    def noise(self):
        if self._custom_noise is not None:
            return self._custom_noise
        return perf_dict_to_noise(self.performance)

    @noise.setter
    def noise(self, noise: NoiseModel | None):
        if self.type == ProcessorType.PHYSICAL:  # TODO: Not sure this is how we should decide (for example: sim:belenos)
            get_logger().warn("Can't set noise to a physical computer", channel.user)
        else:
            self._custom_noise = noise

    @property
    def specs(self) -> PlatformSpecs:
        return self._specs

    @property
    def performance(self):
        return self._perfs

    @property
    def available_parameters(self) -> dict[str, str]:
        return self._specs.parameters

    def validate_single(self, computation: Computation) -> None:
        super().validate_single(computation)
        self.check_min_detected_photons_filter(computation)

    @staticmethod
    def check_min_detected_photons_filter(computation: Computation) -> None:
        # TODO: if we have an iterator, the min_photons_filter can be set only by each iteration
        if computation.experiment.min_photons_filter is None:
            raise ValueError("The value of min_detected_photons is not set."
                             " Use the method experiment.min_detected_photons_filter(value).")

    def _handle_iterator(self, comp: Computation | ComputationIterator, emts: list[AbstractMitigation] = None) -> tuple[Computation, list[AbstractMitigation | ComputationIterator]]:
        # Avoids sending separate jobs if there is an Iterator but no local mitigations
        if comp.command.apply_emt:
            emts = emts if emts is not None else self._error_mitigations
        else:
            emts = []
        if not len(emts):
            return comp, []

        return super()._handle_iterator(comp, emts)

    def _execute_command(self, computation: Computation, progress_cb: ProgressCallback = None) -> dict:
        async_getter = self._execute_single_async(computation)
        # TODO: find a way to use the progress callback in load_async_result or the wait function
        return self._load_async_result(async_getter)

    def _execute_command_async(self, computation: Computation) -> int:
        payload = self.prepare_payload(computation)
        while not self._communication_layer.is_available:
            time.sleep(1)
        return self._communication_layer.send(payload)

    def prepare_payload(self, computation: Computation) -> dict:
        # if self.specs.perceval_version < 1.2.0:
        #     return self._prepare_old_payload(computation)

        # TODO: call a new PayloadGenerator
        payload: dict = {"computation": computation,
                         "mitigations": self._remote_mitigations}
        if len(self._parameters):
            payload["parameters"] = self._parameters
        return payload

    # Only the most basic features should exist here - More advanced features can be added for specific remotes
    def _load_async_result(self, async_getter: AsyncGetter) -> dict:
        while not self.is_complete(async_getter):
            time.sleep(1)
        res = self._communication_layer.get_results(async_getter)

        # if ancient format (with converters):
        #      return convert(res)

        return res

    def is_complete(self, async_getter: AsyncGetter) -> bool:
        # TODO: make this better
        return self._communication_layer.get_status(async_getter) in [RunningStatus.SUCCESS, RunningStatus.ERROR, RunningStatus.CANCELED]

    def cancel(self, async_getter: AsyncGetter) -> None:
        self._communication_layer.cancel(async_getter)

    @property
    def is_remote(self) -> bool:
        return True

    @property
    def type(self):
        return self._specs.type

    def _compute_sample_of_interest_probability(self, computation: Computation | ComputationIterator, param_values: dict = None) -> float:
        # Simulation with a noisy source (only losses)
        computation.validate()

        computation = deepcopy(computation)
        computation.command = CommandFactory.probs

        nm = copy(self.noise)
        nm.g2 = 0
        nm.indistinguishability = 1

        if isinstance(computation, ComputationIterator):
            exp = computation.base_computation.experiment

            # TODO: make a better interface and remove this line + Test if this is useful
            computation._parameter_iterator._experiment = exp
        else:
            exp = computation.experiment

        n = exp.input_state.n
        photon_filter = n
        if exp.min_photons_filter is not None:
            photon_filter = exp.min_photons_filter + sum(exp.heralds.values())
            if photon_filter > n:
                return 0
        if photon_filter < 2:
            return 1

        if param_values is not None:
            params = exp.get_circuit_parameters()
            for param_name, value in param_values.items():
                if param_name in params:
                    params[param_name].set_value(value)

        # Remove all selection
        exp.min_detected_photons_filter(1)
        exp.set_postselection(PostSelect())
        while len(exp.in_heralds):
            m = next(iter(exp.in_heralds))
            exp.remove_port(m, PortLocation.INPUT)
        while len(exp.heralds):
            m = next(iter(exp.heralds))
            exp.remove_port(m, PortLocation.OUTPUT)

        archi = self.specs.architecture
        if archi is not None:
            for m in range(exp.circuit_size):
                exp.add(m, archi.detectors[m])

        lc = SimulatedComputer("SLOS")  # TODO: replace by "best" when available
        lc.noise = nm
        # TODO: how to get default mitigations ?
        lc._error_mitigations = self._error_mitigations + self._remote_mitigations  # TODO: make and use interface

        probs = lc.execute(computation)
        p_above_filter_ns = 0
        for state, prob in probs['results'].items():
            if state.n >= photon_filter:
                p_above_filter_ns += prob
        return p_above_filter_ns

    def estimate_required_shots(self, computation: Computation | ComputationIterator, nsamples: int, param_values: dict = None) -> int:
        """
        Compute an estimate number of required shots given the platform and the user request.
        The circuit, input state, minimum photon filter, and error mitigations are taken into account.

        :param computation: The computation that will be sent with unknown number of samples
        :param nsamples: Number of expected samples of interest
        :param param_values: Key/value pairs for variable parameters inside the circuit. All parameters need to be fixed
            for this computation to run.
        :return: Estimate of the number of shots the user needs to acquire enough samples of interest
        """
        p_interest = self._compute_sample_of_interest_probability(computation, param_values=param_values)
        if p_interest == 0:
            return None
        return round(nsamples / p_interest)

    def estimate_expected_samples(self, computation: Computation | ComputationIterator, nshots: int, param_values: dict = None) -> int:
        """
        Compute an estimate number of samples the user can expect given the platform and the user request.
        The circuit, input state, minimum photon filter, and error mitigations are taken into account.

        :param computation: The computation that will be sent with unknown number of shots
        :param nshots: Number of shots the user is willing to consume
        :param param_values: Key/value pairs for variable parameters inside the circuit. All parameters need to be fixed
            for this computation to run.
        :return: Estimate of the number of samples of interest the user can expect back
        """
        p_interest = self._compute_sample_of_interest_probability(computation, param_values=param_values)
        return round(nshots * p_interest)
