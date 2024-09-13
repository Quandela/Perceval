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
import time
from unittest.mock import patch
from perceval.runtime import JobGroup, RemoteJob
from perceval.components import catalog
from perceval.algorithm import Sampler
from perceval.utils import BasicState, get_logger
from _mock_rpc_handler import get_rpc_handler

TEST_JG_NAME = 'UnitTest_Job_Group'

def test_job_group_creation():
    # create a new job group
    jgroup = JobGroup(TEST_JG_NAME)
    assert jgroup.name == TEST_JG_NAME
    assert len(jgroup.job_group_data) == 0

    datetime_created = jgroup._group_info['created_date']
    datetime_modified = jgroup._group_info['modified_date']
    assert datetime_created == datetime_modified

    # check file now exists
    jg_list = JobGroup.list_saved_job_groups()
    assert any(TEST_JG_NAME in file for file in jg_list)

    # Delete the created file
    JobGroup.delete_job_group(jgroup._file_name)

def test_job_group_load_existing():

    # create a new job group
    jgroup = JobGroup(TEST_JG_NAME)
    datetime_created = jgroup._group_info['created_date']
    datetime_modified = jgroup._group_info['modified_date']

    assert datetime_created == datetime_modified

    # check the jo group exists
    jg_list = JobGroup.list_saved_job_groups()
    assert any(TEST_JG_NAME in file for file in jg_list)

    # set a delay
    time.sleep(1)
    jgroup = JobGroup(TEST_JG_NAME)  # load existing

    datetime_created = jgroup._group_info['created_date']
    datetime_modified = jgroup._group_info['modified_date']
    assert datetime_created < datetime_modified

    # Delete the created file
    JobGroup.delete_job_group(jgroup._file_name)


def test_reject_non_remote_job():
    # creating a local job - sampling
    p = catalog["postprocessed cnot"].build_processor()
    p.with_input(BasicState([0, 1, 0, 1]))
    sampler = Sampler(p)
    local_job = sampler.sample_count

    jgroup = JobGroup(TEST_JG_NAME)
    with pytest.raises(TypeError):
        jgroup.add(local_job)

    # Delete the created file
    JobGroup.delete_job_group(jgroup._file_name)


@patch.object(get_logger(), "warn")
def test_add_remote_to_group(mock_warn, requests_mock):
    remote_job = RemoteJob({}, get_rpc_handler(requests_mock), 'a_remote_job')

    jgroup = JobGroup(TEST_JG_NAME)
    for _ in range(10):
        jgroup.add(remote_job)

    assert len(jgroup.job_group_data) == 10

    # Delete the created file
    JobGroup.delete_job_group(jgroup._file_name)
