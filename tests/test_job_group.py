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

import json
import pytest
import requests
from unittest.mock import MagicMock, patch

import responses

from perceval.runtime import JobGroup, RemoteJob, RunningStatus
from perceval.runtime.rpc_handler import RPCHandler
from perceval.components import catalog
from perceval.algorithm import Sampler
from perceval.utils import BasicState
from _mock_rpc_handler import RPCHandlerResponsesBuilder, CloudEndpoint

TEST_JG_NAME = 'UnitTest_Job_Group'

TOKEN = "test_token"
PLATFORM_NAME = "sim:test"
URL = "https://test"

RPC_HANDLER = RPCHandler(PLATFORM_NAME, URL, TOKEN)


@patch.object(JobGroup._PERSISTENT_DATA, 'write_file')
def test_init(mock_write_file):
    jg = JobGroup(TEST_JG_NAME)
    assert jg.name == TEST_JG_NAME
    assert len(jg) == 0  # empty job group
    assert mock_write_file.call_count == 1


@patch.object(JobGroup._PERSISTENT_DATA, 'write_file')
def test_load(mock_write_file: MagicMock):
    jg = JobGroup(TEST_JG_NAME)

    remote_job_dict = {
        'id': None,
        'status': None,
        'body': {
            'payload': {'job_context': None},
            'job_name': "my_job"},
        'metadata': {
            'headers': RPC_HANDLER.headers,
            'platform': RPC_HANDLER.name,
            'url': RPC_HANDLER.url}
    }

    jg_dict = {
        'created_date': '20250219_103020',
        'modified_date': '20250219_103020',
        'job_group_data': [remote_job_dict, remote_job_dict]}

    jg._deserialize(jg_dict)
    jg._write_to_file()

    last_saved_jg_dict = json.loads(mock_write_file.call_args_list[-1][0][1])

    for key, value in last_saved_jg_dict.items():
        if key == 'modified_date':
            assert jg_dict[key] < value
        else:
            assert jg_dict[key] == value


@patch.object(JobGroup._PERSISTENT_DATA, 'write_file')
def test_add(mock_write_file):
    RPCHandlerResponsesBuilder(RPC_HANDLER)
    job_name = "remote_job_"
    jg = JobGroup(TEST_JG_NAME)

    expected_write_call_count = 1
    assert mock_write_file.call_count == expected_write_call_count

    for i in range(10):
        jg.add(RemoteJob({'payload': {}}, RPC_HANDLER, job_name + str(i)))
        expected_write_call_count += 1
        assert mock_write_file.call_count == expected_write_call_count

    assert len(jg) == 10

    remote_job_dict = {
        'id': None,
        'status': None,
        'body': {
            'payload': {'job_context': None},
            'job_name': job_name},
        'metadata': {
            'headers': RPC_HANDLER.headers,
            'platform': RPC_HANDLER.name,
            'url': RPC_HANDLER.url}
    }

    for i, job_info in enumerate(jg._serialize()['job_group_data']):
        remote_job_dict['body']['job_name'] = job_name + str(i)
        assert job_info == remote_job_dict

    assert mock_write_file.call_count == expected_write_call_count


@patch.object(JobGroup._PERSISTENT_DATA, 'write_file')
def test_add_errors(mock_write_file):
    # creating a local job - sampling
    p = catalog["postprocessed cnot"].build_processor()
    p.with_input(BasicState([0, 1, 0, 1]))
    sampler = Sampler(p)

    jg = JobGroup(TEST_JG_NAME)
    assert mock_write_file.call_count == 1

    with pytest.raises(TypeError):
        jg.add(sampler.sample_count)

    assert mock_write_file.call_count == 1


@patch.object(JobGroup._PERSISTENT_DATA, 'write_file')
def test_classic_run(mock_write_file):
    RPCHandlerResponsesBuilder(RPC_HANDLER)
    rj_nmb = 2

    jg = JobGroup(TEST_JG_NAME)

    expected_write_call_count = 1
    assert mock_write_file.call_count == expected_write_call_count

    for _ in range(rj_nmb):
        jg.add(RemoteJob({'payload': {}}, RPC_HANDLER, 'my_remote_job'))
        expected_write_call_count += 1

    assert mock_write_file.call_count == expected_write_call_count
    assert len(jg) == rj_nmb
    assert len(responses.calls) == 0

    group_progress = jg.progress()

    # no calls or write since jobs have not been sent
    assert len(responses.calls) == 0
    assert mock_write_file.call_count == expected_write_call_count

    assert group_progress == {'Total': rj_nmb,
                              'Finished': [0, {'successful': 0, 'unsuccessful': 0}],
                              'Unfinished': [rj_nmb, {'sent': 0, 'not sent': rj_nmb}]}

    # Running jobs
    jg.run_parallel()
    expected_write_call_count += rj_nmb

    assert len(responses.calls) == rj_nmb
    assert all([CloudEndpoint.from_response(call.response) == CloudEndpoint.CreateJob for call in responses.calls])
    assert mock_write_file.call_count == expected_write_call_count

    group_progress = jg.progress()
    expected_write_call_count += rj_nmb

    assert len(responses.calls) == rj_nmb * 2
    assert all([CloudEndpoint.from_response(call.response) ==
               CloudEndpoint.JobStatus for call in responses.calls[rj_nmb:]])
    assert mock_write_file.call_count == expected_write_call_count

    assert group_progress == {'Total': rj_nmb,
                              'Finished': [rj_nmb, {'successful': rj_nmb, 'unsuccessful': 0}],
                              'Unfinished': [0, {'sent': 0, 'not sent': 0}]}

    for _ in range(rj_nmb):
        jg.add(RemoteJob({'payload': {}}, RPC_HANDLER, 'my_remote_job'))
        expected_write_call_count += 1

    assert mock_write_file.call_count == expected_write_call_count

    group_progress = jg.progress()

    assert len(responses.calls) == rj_nmb * 2
    assert mock_write_file.call_count == expected_write_call_count

    current_group_progress = {'Total': rj_nmb*2,
                              'Finished': [rj_nmb, {'successful': rj_nmb, 'unsuccessful': 0}],
                              'Unfinished': [rj_nmb, {'sent': 0, 'not sent': rj_nmb}]}

    assert group_progress == current_group_progress

    assert mock_write_file.call_count == expected_write_call_count

    # Test complex load

    new_jg = JobGroup(TEST_JG_NAME)
    expected_write_call_count += 1
    assert mock_write_file.call_count == expected_write_call_count

    new_jg._deserialize(jg._serialize())

    # No call on load
    assert len(responses.calls) == rj_nmb * 2
    assert mock_write_file.call_count == expected_write_call_count

    group_progress = jg.progress()

    assert group_progress == current_group_progress

    # No call on load
    assert len(responses.calls) == rj_nmb * 2
    assert mock_write_file.call_count == expected_write_call_count


@patch.object(JobGroup._PERSISTENT_DATA, 'write_file')
def test_save_on_error(mock_write_file):
    rpc_handler_responses_builder = RPCHandlerResponsesBuilder(RPC_HANDLER)
    rpc_handler_responses_builder.set_job_status_sequence([RunningStatus.SUCCESS, None])

    for method_idx in range(2):
        jg = JobGroup(TEST_JG_NAME)

        for _ in range(2):
            jg.add(RemoteJob({'payload': {}}, RPC_HANDLER, 'my_remote_job'))

        with pytest.raises(requests.exceptions.HTTPError):
            if method_idx == 0:
                jg.run_parallel()
            else:
                jg.run_sequential(0.1)

        last_saved_jg_dict = json.loads(mock_write_file.call_args_list[-1][0][1])
        jg = JobGroup(TEST_JG_NAME)
        jg._deserialize(last_saved_jg_dict)

        remote_jobs = jg.remote_jobs
        assert remote_jobs[0].was_sent
        assert remote_jobs[0].is_success
        assert not remote_jobs[1].was_sent


@patch.object(JobGroup._PERSISTENT_DATA, 'write_file')
def test_run_advance(mock_write_file):
    rpc_handler_responses_builder = RPCHandlerResponsesBuilder(RPC_HANDLER)
    rpc_handler_responses_builder.set_job_status_sequence(
        [RunningStatus.SUCCESS, RunningStatus.WAITING, RunningStatus.ERROR])

    jg = JobGroup(TEST_JG_NAME)

    for _ in range(3):
        jg.add(RemoteJob({'payload': {}}, RPC_HANDLER, 'my_remote_job'))

    jg.run_parallel()

    jg.add(RemoteJob({'payload': {}}, RPC_HANDLER, 'my_remote_job'))

    rpc_handler_responses_builder.set_job_status_sequence([])
    rpc_handler_responses_builder.set_default_job_status(RunningStatus.SUCCESS)

    assert jg.progress() == {'Total': 4,
                             'Finished': [2, {'successful': 1, 'unsuccessful': 1}],
                             'Unfinished': [2, {'sent': 1, 'not sent': 1}]}

    jg.run_parallel()

    assert jg.progress() == {'Total': 4,
                             'Finished': [3, {'successful': 2, 'unsuccessful': 1}],
                             'Unfinished': [1, {'sent': 1, 'not sent': 0}]}


@patch.object(JobGroup._PERSISTENT_DATA, 'write_file')
def test_rerun(mock_write_file):
    rpc_handler_responses_builder = RPCHandlerResponsesBuilder(RPC_HANDLER)
    rpc_handler_responses_builder.set_job_status_sequence(
        [RunningStatus.SUCCESS, RunningStatus.WAITING, RunningStatus.ERROR])

    jg = JobGroup(TEST_JG_NAME)

    for _ in range(3):
        jg.add(RemoteJob({'payload': {}}, RPC_HANDLER, 'my_remote_job'))

    jg.run_parallel()

    jg.add(RemoteJob({'payload': {}}, RPC_HANDLER, 'my_remote_job'))

    rpc_handler_responses_builder.set_job_status_sequence([])
    rpc_handler_responses_builder.set_default_job_status(RunningStatus.SUCCESS)

    assert jg.progress() == {'Total': 4,
                             'Finished': [2, {'successful': 1, 'unsuccessful': 1}],
                             'Unfinished': [2, {'sent': 1, 'not sent': 1}]}

    jg.rerun_failed_parallel(replace_failed_jobs=False)

    assert jg.progress() == {'Total': 5,
                             'Finished': [3, {'successful': 2, 'unsuccessful': 1}],
                             'Unfinished': [2, {'sent': 1, 'not sent': 1}]}

    jg.rerun_failed_parallel(replace_failed_jobs=True)

    assert jg.progress() == {'Total': 5,
                             'Finished': [3, {'successful': 3, 'unsuccessful': 0}],
                             'Unfinished': [2, {'sent': 1, 'not sent': 1}]}

# TODO: add payload well saved/load with complex job (iteration parameter et tout)
