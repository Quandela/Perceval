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

import pytest

from perceval import ContextManager, Detector, Imperfections, DetectionType
from perceval.components import Experiment
from perceval.providers.rpc_based_communication_layer import RPCBasedCommunicationLayer
from perceval.runtime.computation import Computation
from perceval.runtime.error_mitigation import (
    CompilationAveraging,
    DetectorBalancing,
    DistinguishablePhotonMitigation,
    PhotonRecycling,
)
from perceval.runtime.execution import Execution
from perceval.runtime.execution_status import JobStatus, RunningStatus
from perceval.runtime.remote_computer import RemoteComputer, _RemoteGetter
from perceval.runtime.simulated_computer import SimulatedComputer
from perceval.runtime.local_computer import _ThreadedGetter
from perceval.serialization import InputArchive, OutputArchive, Serialization
from perceval.utils import NoiseModel


class _SerializableRPCHandler:
    def __init__(self):
        self.fetch_count = 0

    def fetch_platform_details(self):
        self.fetch_count += 1
        return {
            "status": "available",
            "specs": {
                "available_commands": ["probs"],
                "parameters": {"max_shots": "maximum number of shots"},
            },
            "type": "simulator",
            "perfs": {"transmittance": .75},
        }


Serialization.register_class(_SerializableRPCHandler, ["fetch_count"])


def _round_trip(obj):
    archive = OutputArchive()
    Serialization.serialize(obj, archive)
    return Serialization.deserialize(InputArchive.from_text(archive.to_text()))


def test_job_status_serialization():
    status = JobStatus()
    status._status = RunningStatus.ERROR
    status._init_time_start = 10.
    status._running_time_start = 12.
    status._duration = 3.
    status._completed_time = 15.
    status._running_progress = .75
    status._running_phase = "processing"
    status._stop_message = "failed"

    restored = _round_trip(status)

    assert restored.__dict__ == status.__dict__


@pytest.mark.parametrize(
    "mitigation, expected_members",
    (
        (CompilationAveraging(3, 42), {"repetitions": 3, "starting_seed": 42}),
        (DetectorBalancing(), {}),
        (DistinguishablePhotonMitigation({2: 1}), {"_order": {2: 1}}),
        (PhotonRecycling(), {}),
    ),
)
def test_mitigation_serialization(mitigation, expected_members):
    restored = _round_trip(mitigation)

    assert type(restored) is type(mitigation)
    assert restored.__dict__ == expected_members


def test_remote_getter_serialization():
    getter = _RemoteGetter(42, "remote-job-id")
    getter._results = {"results": [1, 2]}
    getter._status.status = RunningStatus.SUCCESS
    getter._last_status_refresh = 123.

    restored = _round_trip(getter)

    assert restored._remote_id == getter._remote_id
    assert restored._results == getter._results
    assert restored._status.__dict__ == getter._status.__dict__  # We don't have __eq__ for JobStatus
    assert restored._communication_layer == 42
    assert restored._last_status_refresh == getter._last_status_refresh
    assert restored._job_status_errors == 0


def test_simulated_computer_serialization():
    computer = SimulatedComputer("SLOS")
    computer.parameters = {"max_shots": 12}
    computer.mitigations = [DetectorBalancing()]
    computer.noise = NoiseModel(transmittance=.8)
    computer._commands["not_serialized"] = object()
    computer._methods["not_serialized"] = object()

    restored = _round_trip(computer)

    assert restored._backend.name == "SLOS"
    assert restored.parameters == {"max_shots": 12}
    assert type(restored.mitigations[0]) is DetectorBalancing
    assert restored.noise == computer.noise
    assert restored.available_commands == ["probs", "samples", "sample_count"]
    assert set(restored._methods) == {"probs", "samples", "sample_count"}


def test_remote_computer_serialization():
    initial_value = RPCBasedCommunicationLayer.MINIMUM_FETCH_INTERVAL
    with ContextManager(lambda: setattr(RPCBasedCommunicationLayer, "MINIMUM_FETCH_INTERVAL", -1),
                        lambda: setattr(RPCBasedCommunicationLayer, "MINIMUM_FETCH_INTERVAL", initial_value),
                        ):
        computer = RemoteComputer(RPCBasedCommunicationLayer(_SerializableRPCHandler()))
        computer.parameters = {"max_shots": 20}
        computer.mitigations = [PhotonRecycling()]
        computer.noise = NoiseModel(transmittance=.6)
        computer.use_mitigations_remotely = False
        computer._commands["not_serialized"] = object()
        computer._specs.available_commands = ["samples"]
        computer._perfs = {"transmittance": .21}
        computer._available_jobs = 99

        restored = _round_trip(computer)

        assert restored.available_commands == ["probs"]
        assert restored.specs.available_commands == ["probs"]
        assert restored.performance == {"transmittance": .75}
        assert restored.available_jobs == 1
        assert restored.parameters == {"max_shots": 20}
        assert type(restored.mitigations[0]) is PhotonRecycling
        assert restored.noise == computer.noise
        assert restored.use_mitigations_remotely is False


def test_execution_serialization():
    computer = SimulatedComputer("SLOS")
    computation = Computation(computer._commands["probs"], Experiment(2))
    execution = Execution(computation, computer)
    execution._user_cb = lambda *_: None
    execution._getters = [[_RemoteGetter(None, "remote-job-id")]]

    restored = _round_trip(execution)

    assert restored._user_cb is None
    assert len(restored._getters) == 1
    assert len(restored._getters[0]) == 1
    assert type(restored._getters[0][0]) is _RemoteGetter
    assert restored._getters[0][0]._remote_id == "remote-job-id"

    execution._getters = [[_ThreadedGetter(lambda: None)]]

    restored = _round_trip(execution)
    assert len(restored._getters) == 0


def test_imperfections_serialization():
    noise = NoiseModel(transmittance=.6)
    d = Detector.threshold()

    imperfections = Imperfections(noise, [d])

    restored = _round_trip(imperfections)

    assert restored.noise == noise
    assert len(restored.detectors) == 1
    assert restored.detectors[0].type == DetectionType.Threshold
