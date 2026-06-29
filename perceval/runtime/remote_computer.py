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
from copy import deepcopy
from typing import TypeVar, Callable

from .command import Command
from .computation import Computation
from .abstract_computer import AbstractComputer
from .computation_iterator import ComputationIterator
from .platform_specs import PlatformSpecs
from .error_mitigation import AbstractMitigation
from .job_status import JobStatus, RunningStatus
from .simulated_computer import SimulatedComputer
from .async_getter import AsyncGetter
from .payload_generator import PayloadGenerator

from perceval.utils import perf_dict_to_noise, ProgressCallback, NoiseModel, PostSelect
from perceval.utils.logging import channel, get_logger
from perceval.components import PortLocation, Experiment

RemoteId = TypeVar("RemoteId")


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

    @abstractmethod
    def send(self, payload: dict) -> RemoteId:
        pass

    @abstractmethod
    def get_results(self, remote_id: RemoteId) -> dict:
        pass

    @abstractmethod
    def get_job_status(self, remote_id: RemoteId, refresh_errors: int = 0) -> JobStatus | None:
        """
        :param remote_id:
        :param refresh_errors: The number of times in a row where this method returned None
        :return: The Job Status if it was available, None otherwise
        """
        pass

    @abstractmethod
    def get_remote_status(self) -> str:
        pass

    @abstractmethod
    def get_performances(self) -> dict:
        pass

    @abstractmethod
    def get_commands(self) -> list[Command]:
        pass

    @abstractmethod
    def cancel(self, remote_id: RemoteId) -> None:
        pass


class _RemoteGetter(AsyncGetter):
    STATUS_REFRESH_DELAY = 1  # minimum job status refresh period (in s)
    _MAX_ERROR = 5

    def __init__(self, communication_layer: CommunicationLayer, remote_id: RemoteId):
        super().__init__()
        # TODO: Communication layer must NOT be serialized. It should be reinserted back when deserializing a Job
        self._communication_layer = communication_layer  # Not serialized
        self._remote_id = remote_id
        self._last_status_refresh = 0.  # Not serialized
        self._job_status_errors = 0  # Not serialized

    def cancel(self):
        if self.status.status in (RunningStatus.RUNNING, RunningStatus.WAITING, RunningStatus.SUSPENDED):
            get_logger().info(f"Programmatically request job {self._remote_id} cancellation", channel.general)
            self._communication_layer.cancel(self._remote_id)
            self._status.stop_run(RunningStatus.CANCEL_REQUESTED, 'Cancellation requested by user')
        else:
            raise RuntimeError('Job is not waiting or running, cannot cancel it')

    def _update_status(self) -> None:
        if self._status.completed:  # static status - retrieving from the cloud unnecessary.
            return

        now = time.time()
        if now - self._last_status_refresh > self.STATUS_REFRESH_DELAY:
            self._previous_status_refresh = now
            status = self._communication_layer.get_job_status(self._remote_id, self._job_status_errors)
            if status is not None:
                self._job_status_errors = 0
                self._status.copy_from(status)
            else:
                self._job_status_errors += 1

    def _get_results(self) -> dict | None:
        if self._results and self.status.completed:
            return self._results
        if not self.status.maybe_completed:
            raise RuntimeError('The job is still running, results are not available yet.')
        self._results = self._communication_layer.get_results(self._remote_id)
        return self._results


class RemoteComputer(AbstractComputer):

    def __init__(self, communication_layer: CommunicationLayer):
        super().__init__()
        self._communication_layer = communication_layer  # cloud_access is the communication layer
        self._commands = {command.name: command for command in communication_layer.get_commands()}
        self._specs = communication_layer.get_specs()
        self._perfs = communication_layer.get_performances()
        self._custom_noise: NoiseModel | None = None
        self.use_mitigations_remotely: bool = True  # TODO: detect if the target supports mitigations ?
        # TODO: how to get default mitigations ?

    @property
    def noise(self):
        if self._custom_noise is not None:
            return self._custom_noise
        return perf_dict_to_noise(self.performance)

    @noise.setter
    def noise(self, noise: NoiseModel | None):
        self._custom_noise = noise

    def _get_local_mitigations(self) -> list[AbstractMitigation]:
        return [] if self.use_mitigations_remotely else super()._get_local_mitigations()

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
        self.check_experiment(computation.experiment)

        params = computation.parameters
        if "max_samples" in params and "max_shots" in params:
            if params["max_samples"] > params["max_shots"]:
                get_logger().warn(f"Lowered 'max_samples' from user defined value ({params['max_samples']}) to"
                                  f" 'max_shots' value ({params['max_shots']}) for consistency.",
                                  channel.user)
                params["max_samples"] = params["max_shots"]

    @staticmethod
    def check_min_detected_photons_filter(experiment: Experiment) -> None:
        # TODO: if we have an iterator, the min_photons_filter can be set only by each iteration
        if experiment.min_photons_filter is None:
            raise ValueError("The value of min_detected_photons is not set."
                             " Use the method experiment.min_detected_photons_filter(value).")

    def check_experiment(self, experiment: Experiment) -> None:
        self.check_min_detected_photons_filter(experiment)

        constraints = self.specs.constraints
        if constraints:
            input_state = experiment.input_state
            n_heralds = sum(experiment.in_heralds.values())
            n_photons = input_state.n + n_heralds
            # Checks on state
            if 'max_photon_count' in constraints and n_photons > constraints['max_photon_count']:
                raise RuntimeError(
                    f"Too many photons in input state ({input_state.n} + {n_heralds} heralds > {constraints['max_photon_count']})")
            if 'min_photon_count' in constraints and n_photons < constraints['min_photon_count']:
                raise RuntimeError(
                    f"Not enough photons in input state ({n_photons} < {constraints['min_photon_count']})")
            if ('support_multi_photon' in constraints and not constraints['support_multi_photon']
                    and not all(mode_photon_cnt <= 1 for mode_photon_cnt in input_state)):
                raise RuntimeError(f"Input state ({input_state}) is not permitted."
                                   " QPU/QPU simulators doesn't accept more than 1 photon per mode")

            # Checks on circuit
            m = experiment.circuit_size
            if 'max_mode_count' in constraints and m > constraints['max_mode_count']:
                raise RuntimeError(f"Circuit too big ({m} modes > {constraints['max_mode_count']})")
            if 'min_mode_count' in constraints and m < constraints['min_mode_count']:
                raise RuntimeError(f"Circuit too small ({m} < {constraints['min_mode_count']})")

    def _handle_iterator(self, comp: Computation | ComputationIterator, out: dict | None)\
            -> tuple[dict, Callable[[dict], None]]:
        if out is None:
            out = dict()

        # Avoids sending separate jobs if there is an Iterator but no local mitigations
        if isinstance(comp, ComputationIterator) and len(self._get_local_mitigations()) > 0:
            return out, comp.make_inserter(out)

        return out, lambda res: out.update(res)

    def extend_computation_keep_original(self, computation: Computation | ComputationIterator) -> list[tuple[list[Computation], Computation]]:
        if len(self._get_local_mitigations()) > 0:
            return super().extend_computation_keep_original(computation)
        else:
            # Avoids sending separate jobs if there is an Iterator but no local mitigations
            # This is doable here because execute_command was made so that it supports ComputationIterator
            return [([computation], computation)]

    def extend_computation(self, computation: Computation | ComputationIterator) -> list[list[Computation]]:
        if len(self._get_local_mitigations()) > 0:
            return super().extend_computation(computation)
        else:
            # Avoids sending separate jobs if there is an Iterator but no local mitigations
            # This is doable here because execute_command was made so that it supports ComputationIterator
            return [[computation]]

    def _execute_command(self, computation: Computation, progress_cb: ProgressCallback = None) -> dict:
        async_getter = self._execute_single_async(computation)
        # TODO: use the progress callback in the wait function
        while not async_getter.is_complete:
            time.sleep(1)
        return async_getter.get_results()

    def _execute_command_async(self, computation: Computation) -> _RemoteGetter:
        # Subclasses may implement something here to ask for availability before sending to the cloud
        payload = self.prepare_payload(computation)
        return _RemoteGetter(self._communication_layer, self._communication_layer.send(payload))

    def prepare_payload(self, computation: Computation) -> dict:
        if self._error_mitigations is not None:
            if self.use_mitigations_remotely:
                remote_mitigations = self._error_mitigations
            else:
                remote_mitigations = []
        else:
            remote_mitigations = None  # Apply default mitigations

        return PayloadGenerator.from_computation(computation,
                                                 remote_mitigations,
                                                 self._parameters,
                                                 self._custom_noise)

    @property
    def is_remote(self) -> bool:
        return True

    @property
    def type(self):
        return self._specs.type

    def _estimate_sample_probability(self, computation: Computation | ComputationIterator, param_values: dict = None) -> float:
        # Simulation with a noisy source (only losses)
        computation.validate()

        lc = SimulatedComputer("SLOS")  # TODO: replace by "best" when available

        computation = deepcopy(computation)
        computation.command = lc.get_command("probs")

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

        nm = deepcopy(self.noise)
        nm.g2 = 0
        nm.indistinguishability = 1
        lc.noise = nm
        # TODO: how to get default mitigations ?
        lc.mitigations = self._error_mitigations

        probs = lc.execute(computation)
        p_above_filter_ns = 0
        for state, prob in probs['results'].items():
            if state.n >= photon_filter:
                p_above_filter_ns += prob
        return p_above_filter_ns

    def estimate_required_shots(self, computation: Computation | ComputationIterator, nsamples: int, param_values: dict = None) -> int | None:
        """
        Compute an estimate number of required shots given the platform and the user request.
        The circuit, input state, minimum photon filter, and error mitigations are taken into account.

        :param computation: The computation that will be sent with unknown number of samples
        :param nsamples: Number of expected samples of interest
        :param param_values: Key/value pairs for variable parameters inside the circuit. All parameters need to be fixed
            for this computation to run.
        :return: Estimate of the number of shots the user needs to acquire enough samples of interest,
            or None if no sample of interest can be acquired
        """
        p_interest = self._estimate_sample_probability(computation, param_values=param_values)
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
        p_interest = self._estimate_sample_probability(computation, param_values=param_values)
        return round(nshots * p_interest)
