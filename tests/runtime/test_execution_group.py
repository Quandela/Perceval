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
from unittest.mock import patch

import pytest
import tqdm
import contextlib
import shutil

import perceval as pcvl
from perceval import RunningStatus, RemoteComputer, FockState
from perceval.components import Experiment
from perceval.runtime import Computation, Execution, ExecutionGroup, SimulatedComputer
from perceval.providers.quandela.rpc_handler import RPCHandler
from perceval.providers.rpc_based_communication_layer import RPCBasedCommunicationLayer

from ..providers.quandela._mock_rpc_handler import RPCHandlerResponsesBuilder

from .test_execution import execution

TOKEN = "test_token"
PLATFORM_NAME = "sim:test"
URL = "https://test"

GROUP_TEST_NAME = "_test_group"
GROUP_TEST_PATH = "./_test_group_56784"  # Avoid any existing path, as it will be deleted

RPC_HANDLER = RPCHandler(PLATFORM_NAME, URL, TOKEN)

INITIAL_REFRESH_DELAY = ExecutionGroup.STATUS_REFRESH_DELAY


def get_group(name = GROUP_TEST_NAME, path = "./_test_execution_groups"):
    return ExecutionGroup(name, path)

@contextlib.contextmanager
def group_context(name = GROUP_TEST_NAME, path = "./_test_execution_groups"):
    group = get_group(name, path)
    try:
        yield group
    finally:
        shutil.rmtree(path, ignore_errors=True)
        ExecutionGroup.STATUS_REFRESH_DELAY = INITIAL_REFRESH_DELAY


def _execution(name="execution"):
    computer = SimulatedComputer("SLOS")
    exp = Experiment(2)
    exp.with_input(FockState([1, 0]))
    computation = Computation(computer.get_command("probs"), exp)
    computation.job_name = name
    return Execution(computation, computer)


def test_persists_and_loads():
    with group_context() as group:
        execution = _execution()
        group.add(execution)

        restored = get_group()

        assert len(restored) == 1
        assert restored[0].name == execution.name
        assert restored.progress() == {
            "Total": 1,
            "Finished": [0, {"successful": 0, "unsuccessful": 0}],
            "Unfinished": [1, {"sent": 0, "not sent": 1}],
        }


def test_add_wrong_arguments():
    with group_context() as group:
        with pytest.raises(TypeError, match="Only an Execution"):
            group.add(object())

        execution = _execution()
        group.add(execution)
        with pytest.raises(ValueError, match="Duplicate"):
            group.add(execution)


@patch.object(pcvl.PersistentData, 'write_file')
@patch.object(tqdm.tqdm, "display")
def test_classic_run(_, mock_write_file):
    with group_context() as eg:
        ExecutionGroup.STATUS_REFRESH_DELAY = 0

        rpc_handler_responses_builder = RPCHandlerResponsesBuilder(RPC_HANDLER)
        exec_nmb = 2

        expected_write_call_count = 1
        assert mock_write_file.call_count == expected_write_call_count

        for _ in range(exec_nmb):
            eg.add(_execution())
            expected_write_call_count += 1

        assert mock_write_file.call_count == expected_write_call_count
        assert len(eg) == exec_nmb

        group_progress = eg.progress()

        # no write since jobs have not been sent
        assert mock_write_file.call_count == expected_write_call_count

        assert group_progress == {'Total': exec_nmb,
                                  'Finished': [0, {'successful': 0, 'unsuccessful': 0}],
                                  'Unfinished': [exec_nmb, {'sent': 0, 'not sent': exec_nmb}]}

        # Running jobs
        rpc_handler_responses_builder.set_job_availability_count(2)

        eg.run_sequential(0)
        expected_write_call_count += 2 * exec_nmb

        assert mock_write_file.call_count == expected_write_call_count
        assert eg[0].job_group_name == eg.name

        group_progress = eg.progress()

        assert mock_write_file.call_count == expected_write_call_count
        assert group_progress == {'Total': exec_nmb,
                                  'Finished': [exec_nmb, {'successful': exec_nmb, 'unsuccessful': 0}],
                                  'Unfinished': [0, {'sent': 0, 'not sent': 0}]}

        for _ in range(exec_nmb):
            eg.add(_execution())
            expected_write_call_count += 1

        assert mock_write_file.call_count == expected_write_call_count

        group_progress = eg.progress()

        assert mock_write_file.call_count == expected_write_call_count

        current_group_progress = {'Total': exec_nmb*2,
                                  'Finished': [exec_nmb, {'successful': exec_nmb, 'unsuccessful': 0}],
                                  'Unfinished': [exec_nmb, {'sent': 0, 'not sent': exec_nmb}]}

        assert group_progress == current_group_progress

        assert mock_write_file.call_count == expected_write_call_count

        # Test complex load
        _ = get_group()
        expected_write_call_count += 1
        assert mock_write_file.call_count == expected_write_call_count


@patch.object(tqdm.tqdm, "display")
def test_run_advance(_):
    with group_context() as eg:
        rpc_handler_responses_builder = RPCHandlerResponsesBuilder(RPC_HANDLER)
        rpc_handler_responses_builder.set_job_status_sequence(
            [RunningStatus.SUCCESS, RunningStatus.ERROR, RunningStatus.ERROR])

        computer = RemoteComputer(RPCBasedCommunicationLayer(RPC_HANDLER))
        experiment = Experiment(2)
        experiment.min_detected_photons_filter(1)
        experiment.with_input(FockState([1, 0]))
        computation = Computation(computer.get_command("probs"), experiment)
        execution = Execution(computation, computer)

        for i in range(3):
            eg.add(execution.clone(), max_shots = i)

        eg.run_parallel()

        assert rpc_handler_responses_builder.last_payload.get("job_group_name") == GROUP_TEST_NAME
        assert rpc_handler_responses_builder.last_payload["payload"].get("max_shots") == 2

        eg.add(execution.clone(), max_shots = 1000)

        rpc_handler_responses_builder.set_job_status_sequence([])
        rpc_handler_responses_builder.set_default_job_status(RunningStatus.SUCCESS)

        assert eg.progress() == {'Total': 4,
                                 'Finished': [3, {'successful': 1, 'unsuccessful': 2}],
                                 'Unfinished': [1, {'sent': 0, 'not sent': 1}]}

        eg.run_parallel()

        assert eg.progress() == {'Total': 4,
                                 'Finished': [4, {'successful': 2, 'unsuccessful': 2}],
                                 'Unfinished': [0, {'sent': 0, 'not sent': 0}]}


def test_cancel_all(execution):
    with group_context() as eg:
        period = 0.03

        for i in range(13):
            eg.add(execution.clone(), n = 5, period = 0. if i < 8 else period)

        eg.launch_async_executions()
        time.sleep(period / 5)  # Give the time for the first threads to finish

        # Unfinished jobs are required in order to cancel_all() doing something
        assert eg.progress() == {'Total': 13,
                                 'Finished': [8, {'successful': 8, 'unsuccessful': 0}],
                                 'Unfinished': [5, {'sent': 5, 'not sent': 0}]}

        eg.cancel_all()

        time.sleep(period + period / 5)  # Give the time for all threads to reach the callback

        assert eg.progress() == {'Total': 13,
                                 'Finished': [13, {'successful': 8, 'unsuccessful': 5}],
                                 'Unfinished': [0, {'sent': 0, 'not sent': 0}]}


def test_separate_folders(execution):
    with group_context() as eg:
        execution = _execution()
        eg.add(execution)

        with group_context(path=GROUP_TEST_PATH + "Test") as eg2:
            assert len(eg2) == 0


def test_separate_names(execution):
    with group_context() as eg:
        execution = _execution()
        eg.add(execution)

        with group_context(name=GROUP_TEST_NAME + "Test") as eg2:
            assert len(eg2) == 0
