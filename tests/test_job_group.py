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
from perceval.runtime import JobGroup, RemoteJob
from perceval.components import catalog
from perceval.algorithm import Sampler
from perceval.utils import BasicState, get_logger
from _mock_rpc_handler import get_rpc_handler

TEST_JG_NAME = 'UnitTest_Job_Group'

@patch.object(JobGroup, '_write_job_group_to_disk')
def test_job_group_creation(mock_write_file):
    jgroup = JobGroup(TEST_JG_NAME)
    assert jgroup.name == TEST_JG_NAME
    assert len(jgroup.list_remote_jobs) == 0  # empty job group

    assert jgroup._group_info['created_date'] == jgroup._group_info['modified_date']

    # assert mock methods called
    mock_write_file.assert_called_once()

@patch.object(JobGroup, '_write_job_group_to_disk')
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

@patch.object(JobGroup, 'list_existing')
@patch.object(JobGroup, '_read_job_group_from_disk')
@patch.object(JobGroup, '_write_job_group_to_disk')
@patch.object(get_logger(), "warn")
def test_add_remote_to_group(mock_warn, mock_write_file,
                             mock_read, mock_list, requests_mock):
    mock_rpc = get_rpc_handler(requests_mock)
    remote_job = RemoteJob({}, mock_rpc, 'a_remote_job')

    jgroup = JobGroup(TEST_JG_NAME)
    for _ in range(10):
        jgroup.add(remote_job)

    assert len(jgroup.list_remote_jobs) == 10

    for each_job in jgroup._group_info['job_group_data']:
        assert each_job['id'] is None
        assert each_job['status'] is None
        assert not each_job['body']  # empty mock remote job added - no body

        # check correct metadata stored
        assert each_job['metadata']['headers'] == mock_rpc.headers
        assert each_job['metadata']['platform'] == mock_rpc.fetch_platform_details()['name']
        assert each_job['metadata']['url'] == mock_rpc.url

    # assert mock method calls
    assert mock_write_file.call_count == 11 # 1 creation + 10 add/modify
    assert mock_read.call_count == 10


@patch.object(JobGroup, 'list_existing')
@patch.object(JobGroup, '_read_job_group_from_disk')
@patch.object(JobGroup, '_write_job_group_to_disk')
@patch.object(get_logger(), "warn")
def test_check_progress_group(mock_warn, mock_write_file,
                             mock_read, mock_list, requests_mock):
    mock_rpc = get_rpc_handler(requests_mock)
    remote_job = RemoteJob({}, mock_rpc, 'a_remote_job')

    num_rj_add = 10
    target_status_list = ['not_sent']*num_rj_add

    jgroup = JobGroup(TEST_JG_NAME)
    for _ in range(num_rj_add):
        jgroup.add(remote_job)

    assert len(jgroup.list_remote_jobs) == 10

    group_status_list = jgroup.progress()

    assert group_status_list == target_status_list
