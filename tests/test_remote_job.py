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

from perceval.runtime import RemoteJob, RunningStatus

import pytest

from _mock_rpc_handler import MockRPCHandler, REMOTE_JOB_DURATION, REMOTE_JOB_RESULTS, REMOTE_JOB_CREATION_TIMESTAMP, \
    REMOTE_JOB_START_TIMESTAMP, REMOTE_JOB_NAME


def test_remote_job():
    _FIRST_JOB_NAME = "job name"
    _SECOND_JOB_NAME = "another name"
    rj = RemoteJob({}, MockRPCHandler(), _FIRST_JOB_NAME)
    assert rj.name == _FIRST_JOB_NAME
    rj.name = _SECOND_JOB_NAME
    assert rj.name == _SECOND_JOB_NAME
    with pytest.raises(TypeError):
        rj.name = None
    with pytest.raises(TypeError):
        rj.name = 28
    job_status = rj.status
    assert rj.is_complete == job_status.completed
    assert rj.get_results()['results'] == REMOTE_JOB_RESULTS

    rj.status.status = RunningStatus.UNKNOWN
    with pytest.warns(UserWarning):
        assert rj.get_results()['results'] == REMOTE_JOB_RESULTS

    _TEST_JOB_ID = "any"
    resumed_rj = RemoteJob.from_id(_TEST_JOB_ID, MockRPCHandler())
    assert resumed_rj.get_results()['results'] == REMOTE_JOB_RESULTS
    assert resumed_rj.id == _TEST_JOB_ID
    assert rj.is_complete == job_status.completed
    assert rj.name == REMOTE_JOB_NAME
    assert rj.status.creation_timestamp == REMOTE_JOB_CREATION_TIMESTAMP
    assert rj.status.start_timestamp == REMOTE_JOB_START_TIMESTAMP
    assert rj.status.duration == REMOTE_JOB_DURATION
