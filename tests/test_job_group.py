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
from unittest.mock import patch

import responses

from perceval.runtime import JobGroup, RemoteJob
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

@patch.object(JobGroup, '_write_to_file')
def test_job_group_creation(mock_write_file):
    jgroup = JobGroup(TEST_JG_NAME)
    assert jgroup.name == TEST_JG_NAME
    assert len(jgroup.list_remote_jobs) == 0  # empty job group
    mock_write_file.assert_called_once()

@patch.object(JobGroup, '_write_to_file')
def test_reject_non_remote_job(mock_write_file):
    # creating a local job - sampling
    p = catalog["postprocessed cnot"].build_processor()
    p.with_input(BasicState([0, 1, 0, 1]))
    sampler = Sampler(p)
    local_job = sampler.sample_count

    jgroup = JobGroup(TEST_JG_NAME)
    with pytest.raises(TypeError):
        jgroup.add(local_job)

    # assert mock methods called
    mock_write_file.assert_called_once()


@patch.object(JobGroup, '_write_to_file')
def test_add_remote_to_group(mock_write_file):
    RPCHandlerResponsesBuilder(RPC_HANDLER)

    jgroup = JobGroup(TEST_JG_NAME)
    for _ in range(10):
        jgroup.add(RemoteJob({'payload': {}}, RPC_HANDLER, 'a_remote_job'))

    assert len(jgroup.list_remote_jobs) == 10

    for each_job in jgroup.to_dict()['job_group_data']:
        assert each_job['id'] is None
        assert each_job['status'] is None
        assert each_job['body'] == {'payload': {'job_context': None}, 'job_name': 'a_remote_job'}

        # check correct metadata stored
        assert each_job['metadata']['headers'] == RPC_HANDLER.headers
        assert each_job['metadata']['platform'] == RPC_HANDLER.fetch_platform_details()['name']
        assert each_job['metadata']['url'] == RPC_HANDLER.url

    # assert mock method calls
    assert mock_write_file.call_count == 11  # 1 creation + 10 add/modify


@patch.object(JobGroup, '_write_to_file')
def test_check_group_progress(mock_write_file):
    rpc_handler_responses_builder = RPCHandlerResponsesBuilder(RPC_HANDLER)
    rj_nmb = 10

    jg = JobGroup(TEST_JG_NAME)
    for _ in range(rj_nmb):
        jg.add(RemoteJob({'payload': {}}, RPC_HANDLER, 'a_remote_job'))

    assert len(jg.list_remote_jobs) == rj_nmb
    assert mock_write_file.call_count == rj_nmb + 1

    assert len(responses.calls) == 0

    group_progress = jg.progress()

    assert len(responses.calls) == 0  # no call since jobs have not been sent
    assert mock_write_file.call_count == rj_nmb + 1  # no need to save because no changes

    assert group_progress == {'Total': rj_nmb,
                              'Finished': [0, {'successful': 0, 'unsuccessful': 0}],
                              'Unfinished': [rj_nmb, {'sent': 0, 'not sent': rj_nmb}]}

    jg.run_parallel()

    assert len(responses.calls) == rj_nmb
    assert all([CloudEndpoint.from_response(call.response) == CloudEndpoint.CreateJob for call in responses.calls])
    assert mock_write_file.call_count == 2*rj_nmb + 1  # no need to save because no changes

    group_progress = jg.progress()

    assert len(responses.calls) == rj_nmb * 2
    assert all([CloudEndpoint.from_response(call.response) ==
               CloudEndpoint.JobStatus for call in responses.calls[rj_nmb:]])
    assert mock_write_file.call_count == 3*rj_nmb + 1

    assert group_progress == {'Total': rj_nmb,
                              'Finished': [rj_nmb, {'successful': rj_nmb, 'unsuccessful': 0}],
                              'Unfinished': [0, {'sent': 0, 'not sent': 0}]}

    for _ in range(rj_nmb):
        jg.add(RemoteJob({'payload': {}}, RPC_HANDLER, 'a_remote_job'))
    assert mock_write_file.call_count == 4*rj_nmb + 1

    group_progress = jg.progress()

    assert len(responses.calls) == rj_nmb * 2
    assert mock_write_file.call_count == 4*rj_nmb + 1

    assert group_progress == {'Total': rj_nmb*2,
                              'Finished': [rj_nmb, {'successful': rj_nmb, 'unsuccessful': 0}],
                              'Unfinished': [rj_nmb, {'sent': 0, 'not sent': rj_nmb}]}
