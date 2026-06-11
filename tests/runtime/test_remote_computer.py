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

from perceval import RunningStatus, AbstractComputer, SimulatedComputer, Experiment, FockState, Computation, \
    CommandFactory, BSDistribution, JobStatus
from perceval.runtime.computation_iterator import ComputationIterator
from perceval.runtime.platform_specs import PlatformSpecs
from perceval.runtime.remote_computer import CommunicationLayer, RemoteComputer, RemoteId
from perceval.utils.constants import KEY_NOISE, KEY_PARAMETERS, KEY_COMPUTATION, KEY_MITIGATIONS
from tests._test_utils import assert_bsd_close


class ComputerProxy(CommunicationLayer):

    def __init__(self, computer: AbstractComputer) -> None:
        self.computer = computer

    def get_specs(self) -> PlatformSpecs:
        return self.computer.specs

    def send(self, payload: dict) -> list:
        computation = payload[KEY_COMPUTATION]
        mitigations = payload[KEY_MITIGATIONS]
        self.computer.set_mitigations(mitigations)
        if KEY_PARAMETERS in payload:
            self.computer.set_parameters(payload[KEY_PARAMETERS])
        else:
            self.computer.reset_parameters()
        if KEY_NOISE in payload:
            self.computer.noise = payload[KEY_NOISE]
        return [computation, *self.computer.execute_async(computation)]

    def get_results(self, remote_id: list) -> dict:
        while not all(getter.is_complete for getter in remote_id[-1]):
            time.sleep(0.1)
        return self.computer.get_results(*remote_id)

    def get_job_status(self, remote_id: list, refresh_errors: int = 0) -> JobStatus | None:
        # TODO: account better for progress and times
        for getter in remote_id[-1]:
            status = getter.status
            if not status.completed:
                return status
        return status

    def get_remote_status(self) -> str:
        return "available"

    def get_performances(self) -> dict:
        return self.computer.performance

    def get_commands(self) -> list[str]:
        return self.computer.available_commands

    def cancel(self, remote_id: list) -> None:
        for getter in remote_id[-1]:
            getter.cancel()


def test_remote_computer_basic():
    # Checks that the communication layer is properly used
    local_computer = SimulatedComputer("SLOS")
    remote_computer = RemoteComputer(ComputerProxy(local_computer))

    assert remote_computer.available_commands == local_computer.available_commands
    assert remote_computer.available_parameters == local_computer.available_parameters

    assert remote_computer.is_remote

    assert remote_computer.performance == local_computer.performance
    assert remote_computer.noise == local_computer.noise
    assert remote_computer.type == local_computer.type
    assert remote_computer.specs == local_computer.specs


def test_remote_computer_execute():
    remote_computer = RemoteComputer(ComputerProxy(SimulatedComputer("SLOS")))

    e = Experiment(2)
    e.with_input(FockState([1, 0]))
    e.min_detected_photons_filter(1)

    computation = Computation(CommandFactory.probs, e)
    res = remote_computer.execute(computation)

    assert res["results"] == BSDistribution(FockState([1, 0]))


def test_remote_computer_execute_async():
    remote_computer = RemoteComputer(ComputerProxy(SimulatedComputer("SLOS")))

    e = Experiment(2)
    e.with_input(FockState([1, 0]))
    e.min_detected_photons_filter(1)

    computation = Computation(CommandFactory.probs, e)
    mitigations, noise, getter = remote_computer.execute_async(computation)

    while not getter[0].is_complete:
        time.sleep(0.1)

    res = remote_computer.get_results(computation, mitigations, noise, getter)
    assert res["results"] == BSDistribution(FockState([1, 0]))

    assert getter[0].is_complete


def test_remote_computer_execute_iterator():
    remote_computer = RemoteComputer(ComputerProxy(SimulatedComputer("SLOS")))

    experiment = Experiment(2)
    experiment.min_detected_photons_filter(1)

    computation = Computation(CommandFactory.probs, experiment)
    computation = ComputationIterator(computation)

    computation.add_iteration(input_state=FockState([1, 0]))
    computation.add_iteration(input_state=FockState([0, 1]))

    res = remote_computer.execute(computation)

    assert isinstance(res, dict)
    assert "results_list" in res
    assert len(res["results_list"]) == 2

    assert_bsd_close(res["results_list"][0]["results"], BSDistribution(FockState([1, 0])))
    assert_bsd_close(res["results_list"][1]["results"], BSDistribution(FockState([0, 1])))

    assert "iteration" in res["results_list"][0]
    assert res["results_list"][0]["iteration"] == {"input_state": FockState([1, 0])}
    assert res["results_list"][1]["iteration"] == {"input_state": FockState([0, 1])}


def test_remote_computer_execute_async_iterator():
    remote_computer = RemoteComputer(ComputerProxy(SimulatedComputer("SLOS")))

    experiment = Experiment(2)
    experiment.min_detected_photons_filter(1)

    computation = Computation(CommandFactory.probs, experiment)
    computation = ComputationIterator(computation)

    computation.add_iteration(input_state=FockState([1, 0]))
    computation.add_iteration(input_state=FockState([0, 1]))

    mitigations, noise, getter = remote_computer.execute_async(computation)

    assert len(getter) == 1, "Iterator must not be decomposed when there is no local mitigations"

    while not getter[0].is_complete:
        time.sleep(0.1)

    res = remote_computer.get_results(computation, mitigations, noise, getter)

    assert isinstance(res, dict)
    assert "results_list" in res
    assert len(res["results_list"]) == 2

    assert_bsd_close(res["results_list"][0]["results"], BSDistribution(FockState([1, 0])))
    assert_bsd_close(res["results_list"][1]["results"], BSDistribution(FockState([0, 1])))

    assert "iteration" in res["results_list"][0]
    assert res["results_list"][0]["iteration"] == {"input_state": FockState([1, 0])}
    assert res["results_list"][1]["iteration"] == {"input_state": FockState([0, 1])}
