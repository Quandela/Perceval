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
import random
import time
from copy import copy
from typing import TypeAlias

from perceval import AbstractComputer, SimulatedComputer, Experiment, FockState, Computation, BSDistribution, JobStatus, \
    Unitary, BS, PS, NoiseModel, Circuit, Detector, FFCircuitProvider, Command, P, PayloadGenerator, Execution
from perceval.runtime.computation_iterator import ComputationIterator
from perceval.runtime.platform_specs import PlatformSpecs
from perceval.runtime.remote_computer import CommunicationLayer, RemoteComputer
from tests._test_utils import assert_bsd_close


RemoteId: TypeAlias = Execution


class ComputerProxy(CommunicationLayer):

    def __init__(self, computer: AbstractComputer) -> None:
        self.computer = computer

    def get_specs(self) -> PlatformSpecs:
        return self.computer.specs

    def send(self, payload: dict) -> RemoteId:
        with PayloadGenerator.payload_applier(self.computer, payload):
            computation = PayloadGenerator.get_computation(payload)
            # I'm not sure that payload_applier works well with execute_async, so we make a copy of self.computer
            return Execution(computation, copy(self.computer)).execute_async()

    def get_results(self, remote_id: RemoteId) -> dict:
        while not remote_id.is_complete:
            time.sleep(0.1)
        return remote_id.get_results(allow_partial_results=True)

    def get_job_status(self, remote_id: RemoteId, refresh_errors: int = 0) -> JobStatus | None:
        return remote_id.status

    def get_remote_status(self) -> str:
        return "available"

    def get_performances(self) -> dict:
        return self.computer.performance

    def get_commands(self) -> list[Command]:
        return [self.computer.get_command(command) for command in self.computer.available_commands]

    def cancel(self, remote_id: RemoteId) -> None:
        remote_id.cancel()


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

    computation = Computation(remote_computer.get_command("probs"), e)
    res = remote_computer.execute(computation)

    assert res["results"] == BSDistribution(FockState([1, 0]))


def test_remote_computer_execute_async():
    remote_computer = RemoteComputer(ComputerProxy(SimulatedComputer("SLOS")))

    e = Experiment(2)
    e.with_input(FockState([1, 0]))
    e.min_detected_photons_filter(1)

    computation = Computation(remote_computer.get_command("probs"), e)
    mitigations, noise, getter = remote_computer.execute_async(computation)

    while not getter[0][0].is_complete:
        time.sleep(0.1)

    res = remote_computer.get_results(computation, mitigations, noise, getter)
    assert res["results"] == BSDistribution(FockState([1, 0]))

    assert getter[0][0].is_complete


def test_remote_computer_execute_iterator():
    remote_computer = RemoteComputer(ComputerProxy(SimulatedComputer("SLOS")))

    experiment = Experiment(2)
    experiment.min_detected_photons_filter(1)

    computation = Computation(remote_computer.get_command("probs"), experiment)
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

    computation = Computation(remote_computer.get_command("probs"), experiment)
    computation = ComputationIterator(computation)

    computation.add_iteration(input_state=FockState([1, 0]))
    computation.add_iteration(input_state=FockState([0, 1]))

    mitigations, noise, getter = remote_computer.execute_async(computation)

    assert len(getter) == 1, "Iterator must not be decomposed when there is no local mitigations"
    assert len(getter[0]) == 1, "Iterator must not be decomposed when there is no local mitigations"

    while not getter[0][0].is_complete:
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


def test_shots_estimate_trivial_filter_values():
    e = Experiment()
    e.set_circuit(Unitary.random(10))
    e.with_input(FockState([1]*5 + [0]*5))
    e.min_detected_photons_filter(1)

    ANY_VALUE = random.randint(1000, 9999999999)

    remote_computer = RemoteComputer(ComputerProxy(SimulatedComputer("SLOS")))
    computation = Computation(remote_computer.get_command("probs"), e)

    # with min_detected_photons_filter set to 1, shots and samples are the same
    assert remote_computer.estimate_expected_samples(computation, ANY_VALUE) == ANY_VALUE
    assert remote_computer.estimate_required_shots(computation, ANY_VALUE) == ANY_VALUE

    e.min_detected_photons_filter(0)
    # same with 0
    assert remote_computer.estimate_expected_samples(computation, ANY_VALUE) == ANY_VALUE
    assert remote_computer.estimate_required_shots(computation, ANY_VALUE) == ANY_VALUE

    # with a filter too high, there's no estimate
    e.min_detected_photons_filter(6)  # = input_state.n + 1
    assert remote_computer.estimate_expected_samples(computation, ANY_VALUE) == 0
    assert remote_computer.estimate_required_shots(computation, ANY_VALUE) is None


def test_shots_estimate_regular_use_case():
    computer = SimulatedComputer("SLOS")
    computer.noise = NoiseModel(transmittance=0.06)
    remote_computer = RemoteComputer(ComputerProxy(computer))

    c = BS() // PS(phi=0.2) // BS()
    e = Experiment(c)
    e.with_input(FockState([1, 1]))
    computation = Computation(remote_computer.get_command("probs"), e)
    assert 28 < remote_computer.estimate_expected_samples(computation, 1000) < 32
    assert 32000 < remote_computer.estimate_required_shots(computation, 1000) < 33000


def test_shots_estimate_circuit_with_variables():
    computer = SimulatedComputer("SLOS")
    computer.noise = NoiseModel(transmittance=0.06)
    remote_computer = RemoteComputer(ComputerProxy(computer))

    c = BS() // PS(phi=P("my_phase")) // BS()
    e = Experiment(c)
    e.with_input(FockState([1, 1]))

    computation = Computation(remote_computer.get_command("probs"), e)
    assert 28 < remote_computer.estimate_expected_samples(computation, 1000, {"my_phase": 0.2}) < 32
    assert 32000 < remote_computer.estimate_required_shots(computation, 1000, {"my_phase": 0.2}) < 33000


def test_shots_estimate_feed_forward():
    exp_ff = Experiment(4)
    exp_ff.add(0, BS.H())
    for i in range(2):
        exp_ff.add(i, Detector.pnr())
    ffc = FFCircuitProvider(2, 0, BS.H())
    ffc.add_configuration((0, 1), Circuit(2))
    exp_ff.add(0, ffc)
    exp_ff.with_input(FockState([1, 0, 1, 0]))

    computer = SimulatedComputer("SLOS")
    computer.noise = NoiseModel(transmittance=0.06)
    remote_computer = RemoteComputer(ComputerProxy(computer))
    computation = Computation(remote_computer.get_command("probs"), exp_ff)

    assert 28 < remote_computer.estimate_expected_samples(computation, 1000) < 32
    assert 32000 < remote_computer.estimate_required_shots(computation, 1000) < 33000
