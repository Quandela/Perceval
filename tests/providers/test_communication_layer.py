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

from perceval import ProcessorType, Computation, CommandFactory, Experiment, PayloadGenerator, RunningStatus, JobStatus
from perceval.providers.quandela.rpc_handler import RPCHandler
from perceval.providers.rpc_based_communication_layer import RPCBasedCommunicationLayer
from perceval.runtime.platform_specs import PlatformSpecs
from perceval.serialization import deserialize

from .quandela._mock_rpc_handler import RPCHandlerResponsesBuilder, ARCHITECTURE_PLATFORM_INFO
from .._test_utils import assert_experiment_equals

TOKEN = "test_token"
PLATFORM_NAME = "sim:test"
URL = "https://test"


# Note: Quandela, Kipu and Scaleway Communication layers share 95% of their code through the RPCBasedCommunicationLayer,
# so testing this one should be enough

def test_communication_layer_platform_status():
    rpc_handler = RPCHandler(PLATFORM_NAME, URL, TOKEN)
    RPCHandlerResponsesBuilder(rpc_handler, ARCHITECTURE_PLATFORM_INFO, use_new_platform_details_url=True)

    comm = RPCBasedCommunicationLayer(rpc_handler)

    specs = comm.get_specs()
    assert isinstance(specs, PlatformSpecs)
    expected = PlatformSpecs(ARCHITECTURE_PLATFORM_INFO["specs"])
    expected.type = ProcessorType[ARCHITECTURE_PLATFORM_INFO["type"].upper()]
    expected_architecture = deserialize(expected.pop("architecture"))
    gotten_architecture = specs.pop("architecture")
    assert specs == expected

    assert_experiment_equals(expected_architecture, gotten_architecture)

    perfs = comm.get_performances()
    assert perfs == ARCHITECTURE_PLATFORM_INFO["perfs"]

    platform_status = comm.get_remote_status()
    assert platform_status == ARCHITECTURE_PLATFORM_INFO["status"]

    commands = comm.get_commands()
    assert commands == specs.commands


def test_communication_layer_job():
    expected_running_status = [RunningStatus.RUNNING, RunningStatus.CANCELED, RunningStatus.SUCCESS]

    rpc_handler = RPCHandler(PLATFORM_NAME, URL, TOKEN)
    builder = RPCHandlerResponsesBuilder(rpc_handler,
                               ARCHITECTURE_PLATFORM_INFO,
                               expected_running_status,
                               use_new_platform_details_url=True)
    builder.set_job_status_sequence(expected_running_status)

    comm = RPCBasedCommunicationLayer(rpc_handler)

    computation = Computation(CommandFactory.probs, Experiment())

    with pytest.raises(Exception):
        comm.send(PayloadGenerator.from_computation(computation))

    computation.add_params(max_shots = 10000)

    remote_id = comm.send(PayloadGenerator.from_computation(computation))
    status = comm.get_job_status(remote_id)
    assert isinstance(status, JobStatus)
    assert status.status == expected_running_status[0]
    assert status.progress == 0.5

    comm.cancel(remote_id)  # Just check nothrow, the ResponseBuilder doesn't change the status to canceled by itself

    remote_id = comm.send(PayloadGenerator.from_computation(computation))
    status = comm.get_job_status(remote_id)
    assert isinstance(status, JobStatus)
    assert status.status == expected_running_status[1]

    # Due to the way the ResponseBuilder works, we need to create another job
    remote_id = comm.send(PayloadGenerator.from_computation(computation))
    status = comm.get_job_status(remote_id)
    assert isinstance(status, JobStatus)
    assert status.status == expected_running_status[2]
    assert status.progress == 1.

    res = comm.get_results(remote_id)
    assert "results" in res  # Check this is the correct level of results
    assert "physical_perf" in res
    assert "logical_perf" in res
