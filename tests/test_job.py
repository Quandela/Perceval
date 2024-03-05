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
import perceval as pcvl
from perceval.runtime.job_status import RunningStatus


def quadratic_count_down(n, speed=0.1, progress_callback=None):
    l = []
    for i in range(n):
        time.sleep(speed)
        if progress_callback:
            progress_callback(i / n, "counting %d" % i)
        l.append(i ** 2)
    assert speed >= 0.1
    return l


def test_run_sync_0():
    assert (pcvl.LocalJob(quadratic_count_down, command_param_names=['n', 'speed'])(5) == [0, 1, 4, 9, 16])


def test_run_sync_1():
    job = pcvl.LocalJob(quadratic_count_down, command_param_names=['n', 'speed'])
    assert job.execute_sync(5) == [0, 1, 4, 9, 16]
    assert job.is_complete
    assert job.status.success
    assert job.get_results() == [0, 1, 4, 9, 16]
    # should be ~ 0.5 s
    assert job.status.running_time < 1
    assert job.status.status == RunningStatus.SUCCESS

    job.status.status = RunningStatus.UNKNOWN
    with pytest.warns(UserWarning):
        assert job.get_results() == [0, 1, 4, 9, 16]


def test_run_async():
    job = pcvl.LocalJob(quadratic_count_down, command_param_names=['n', 'speed'])
    assert job.execute_async(5, 0.3) is job
    assert not job.is_complete
    counter = 0
    while not job.is_complete:
        counter += 1
        time.sleep(0.5)
    assert counter > 1
    assert job.status.success
    assert job.status.stop_message is None
    assert job.get_results() == [0, 1, 4, 9, 16]
    assert job.status.progress == 1
    # should be at least 1.5s
    assert job.status.running_time > 1
    assert job.status.status == RunningStatus.SUCCESS


def test_run_async_fail():
    job = pcvl.LocalJob(quadratic_count_down, command_param_names=['n', 'speed'])
    assert job.execute_async(5, 0.01) is job

    with pytest.warns(UserWarning):
        while not job.is_complete:
            time.sleep(1)
    assert not job.status.success
    assert job.status.progress == 0.8
    assert job.status.status == RunningStatus.ERROR
    assert "AssertionError" in job.status.stop_message
    # should be ~0.05 s
    assert job.status.running_time < 0.5

    job.status.status = RunningStatus.UNKNOWN
    with pytest.warns(UserWarning):
        assert job.get_results() == None


def test_run_async_cancel():
    job = pcvl.LocalJob(quadratic_count_down, command_param_names=['n', 'speed'])
    assert job.execute_async(5, 0.3) is job
    job.cancel()
    while job.is_running:
        time.sleep(0.1)
    assert job.status.status == RunningStatus.CANCELED


def test_get_res_run_async():
    u = pcvl.Unitary(pcvl.Matrix.random_unitary(6))  # a random unitary matrix
    bs = pcvl.BasicState("|1,0,1,0,1,0>")  # basic state
    proc = pcvl.Processor("SLOS", u)  # a processor with a circuit formed of random unitary matrix
    proc.with_input(bs)  # setting up the input to the processor
    job = pcvl.algorithm.Sampler(proc).sample_count  # create a sampler job
    job.execute_async(10000)
    while not job.is_complete:
        time.sleep(0.01)

    res_1st_call = job.get_results()
    res_2nd_call = job.get_results()

    assert isinstance(res_1st_call["results"], pcvl.BSCount)
    assert isinstance(res_2nd_call["results"], pcvl.BSCount)

    assert res_1st_call["results"] == res_2nd_call["results"]
    assert res_1st_call["physical_perf"] == res_2nd_call["physical_perf"]
    assert res_1st_call["logical_perf"] == res_2nd_call["logical_perf"]



# ============ Remote jobs ============ #
from perceval.runtime import RemoteJob
import pytest
import time
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
