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
import time


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
    counter = 0
    while not job.is_complete:
        counter += 1
        time.sleep(1)
    assert not job.status.success
    assert job.status.progress == 0.8
    assert job.status.status == RunningStatus.ERROR
    assert "AssertionError" in job.status.stop_message
    # should be ~0.05 s
    assert job.status.running_time < 0.5


def test_run_async_cancel():
    job = pcvl.LocalJob(quadratic_count_down, command_param_names=['n', 'speed'])
    assert job.execute_async(5, 0.3) is job
    job.cancel()
    while job.is_running:
        time.sleep(0.1)
    assert job.status.status == RunningStatus.CANCELED


# ============ Remote jobs ============ #
from perceval.runtime import RemoteJob
from perceval.serialization import serialize
import json
import pytest
import time

_REMOTE_JOB_NAME = "a remote job"
_REMOTE_JOB_DURATION = 5
_REMOTE_JOB_CREATION_TIMESTAMP = 1687883254.77622
_REMOTE_JOB_START_TIMESTAMP = 1687883263.280909


class MockRPCHandler:
    _ARBITRARY_JOB_ID = "ebb1f8ec-0125-474f-9ffc-5178afef4d1a"
    _SLEEP_SEC = 0.2

    def fetch_platform_details(self):
        time.sleep(self._SLEEP_SEC)
        return json.dumps({
            'created_date': 'Mon, 31 Oct 2022 16:54:45 GMT',
            'description': 'Mocked Simulator',
            'id': 'e576e49c-7b1a-470b-5910-c04e406d40f6',
            'jobs': 6687,
            'name': 'mocked:platform',
            'perfs': {},
            'specs': {
                'available_commands': ['probs'], 'connected_input_modes': [0, 2, 4, 6, 8],
                'constraints': {
                    'max_mode_count': 12,
                    'max_photon_count': 5,
                    'min_mode_count': 1,
                    'min_photon_count': 1
                },
                'description': 'Simulator of sim:ascella qpu',
                'detector': 'threshold',
                'parameters': {
                    'HOM': 'indistinguishability value, using HOM model (default 1)',
                    'backend_name': 'name of the backend that will be used for computation (default "SLOS")',
                    'final_mode_number': 'number of modes of the output states. states having a photon on unused modes will be ignored. Useful when using computed circuits (default input_state.m)',
                    'g2': 'g2 value (default 0)',
                    'mode_post_select': 'number of required detected modes to keep a state. (default input_state.n)',
                    'phase_imprecision': 'imprecision on the phase shifter phases (default 0)',
                    'transmittance': 'probability at each pulse that a photon is sent to the system and is detected (default 1)'
                },
            },
            'status': 'available',
            'svg': '',
            'type': 'simulator',
            'waiting_jobs': 0
        })

    def create_job(self, payload):
        time.sleep(self._SLEEP_SEC)
        return self._ARBITRARY_JOB_ID

    def cancel_job(self, job_id: str):
        time.sleep(self._SLEEP_SEC)

    def get_job_status(self, job_id: str):
        time.sleep(self._SLEEP_SEC)
        return {
            "creation_datetime": _REMOTE_JOB_CREATION_TIMESTAMP,
            "duration": _REMOTE_JOB_DURATION,
            "failure_code": None,
            "last_intermediate_results": None,
            "msg": "ok",
            "name": _REMOTE_JOB_NAME,
            "progress": 1.0,
            "progress_message": "Computing phases to apply (step 2)",
            "start_time": _REMOTE_JOB_START_TIMESTAMP,
            "status": "completed",
            "status_message": None
        }

    def get_job_results(self, job_id: str):
        time.sleep(self._SLEEP_SEC)
        return json.dumps(serialize({
            'results': pcvl.BSDistribution({
                pcvl.BasicState([1, 0, 0, 0]): 0.200266,
                pcvl.BasicState([0, 1, 0, 0]): 0.09734,
                pcvl.BasicState([0, 0, 1, 0]): 0.089365,
                pcvl.BasicState([0, 0, 0, 1]): 0.223731,
                pcvl.BasicState([1, 0, 1, 0]): 0.308951
            }),
            'physical_perf': 0.7988443869134395,
            'job_context': {
                'mapping_delta_parameters': {'count': 10000},
                'result_mapping': ['perceval.utils', 'probs_to_sample_count']
            }
        }))


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

    _TEST_JOB_ID = "any"
    resumed_rj = RemoteJob.from_id(_TEST_JOB_ID, MockRPCHandler())
    assert resumed_rj.id == _TEST_JOB_ID
    assert rj.is_complete == job_status.completed
    assert rj.name == _REMOTE_JOB_NAME
    assert rj.status.creation_timestamp == _REMOTE_JOB_CREATION_TIMESTAMP
    assert rj.status.start_timestamp == _REMOTE_JOB_START_TIMESTAMP
    assert rj.status.duration == _REMOTE_JOB_DURATION
