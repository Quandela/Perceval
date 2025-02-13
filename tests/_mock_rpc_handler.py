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
"""module rpc mock handler"""

import time
import json
from urllib.parse import quote_plus
from perceval.utils import BSDistribution, BasicState
from perceval.serialization import serialize
from perceval.runtime.rpc_handler import (
    RPCHandler,
    _ENDPOINT_JOB_CANCEL,
    _ENDPOINT_JOB_CREATE,
    _ENDPOINT_JOB_RESULT,
    _ENDPOINT_JOB_STATUS,
    _ENDPOINT_PLATFORM_DETAILS,
    _ENDPOINT_JOB_RERUN,
    _JOB_ID_KEY
)

REMOTE_JOB_DURATION = 5
REMOTE_JOB_CREATION_TIMESTAMP = 1687883254.77622
REMOTE_JOB_START_TIMESTAMP = 1687883263.280909
REMOTE_JOB_RESULTS = BSDistribution(
    {
        BasicState([1, 0, 0, 0]): 0.200266,
        BasicState([0, 1, 0, 0]): 0.09734,
        BasicState([0, 0, 1, 0]): 0.089365,
        BasicState([0, 0, 0, 1]): 0.223731,
        BasicState([1, 0, 1, 0]): 0.308951,
    }
)


def get_rpc_handler(requests_mock, url='http://test'):
    """return a fake rpc handler"""
    mock = MockRPCHandler(name='mocked:platform', url=url, token='no-token')
    mock.set_mock(requests_mock)
    return mock


class MockRPCHandler(RPCHandler):
    """mock of the rpc handler"""

    _SLEEP_SEC = 0.2
    requests_mock = None

    def set_mock(self, mock):
        """set the mock request"""
        self.requests_mock = mock

    def fetch_platform_details(self):
        time.sleep(self._SLEEP_SEC)
        quot_nam = quote_plus(self.name)
        endpoint = super().build_endpoint(_ENDPOINT_PLATFORM_DETAILS, quot_nam)
        return_json = {
            'created_date': 'Mon, 31 Oct 2022 16:54:45 GMT',
            'description': 'Mocked Simulator',
            'id': 'e576e49c-7b1a-470b-5910-c04e406d40f6',
            'jobs': 6687,
            'name': self.name,
            'perfs': {},
            'specs': {
                'available_commands': ['probs'],
                'connected_input_modes': [0, 2, 4, 6, 8, 10],
                'constraints': {
                    'max_mode_count': 20,
                    'max_photon_count': 6,
                    'min_mode_count': 1,
                    'min_photon_count': 1,
                },
                'description': 'Simulator of sim:altair qpu',
                'detector': 'threshold',
                'parameters': {
                    'HOM': 'indistinguishability value, using HOM model (default 1)',
                    'backend_name': 'name of the backend that will be used for computation (default "SLOS")',
                    'final_mode_number': 'number of modes of the output states. states having a photon on unused modes will be ignored. Useful when using computed circuits (default input_state.m)',
                    'g2': 'g2 value (default 0)',
                    'mode_post_select': 'number of required detected modes to keep a state. (default input_state.n)',
                    'phase_imprecision': 'imprecision on the phase shifter phases (default 0)',
                    'transmittance': 'probability at each pulse that a photon is sent to the system and is detected (default 1)',
                },
            },
            'status': 'available',
            'svg': '',
            'type': 'simulator',
            'waiting_jobs': 0,
        }
        self.requests_mock.get(
            endpoint,
            json=return_json,
        )
        return super().fetch_platform_details()

    def create_job(self, payload):
        time.sleep(self._SLEEP_SEC)
        endpoint = self.build_endpoint(_ENDPOINT_JOB_CREATE)
        arbitrary_job_id = 'ebb1f8ec-0125-474f-9ffc-5178afef4d1a'
        return_json = {_JOB_ID_KEY: arbitrary_job_id}
        self.requests_mock.post(endpoint, json=return_json)
        return super().create_job(payload)

    def cancel_job(self, job_id: str):
        time.sleep(self._SLEEP_SEC)
        endpoint = self.build_endpoint(_ENDPOINT_JOB_CANCEL, job_id)
        self.requests_mock.post(endpoint, json={})
        return super().cancel_job(job_id)

    def rerun_job(self, job_id: str):
        time.sleep(self._SLEEP_SEC)
        endpoint = self.build_endpoint(_ENDPOINT_JOB_RERUN, job_id)
        arbitrary_job_id = '32e74bda-afc5-41d3-88de-ab68f9805cc4'
        return_json = {_JOB_ID_KEY: arbitrary_job_id}
        self.requests_mock.post(endpoint, json=return_json)
        return super().rerun_job(job_id)

    def get_job_status(self, job_id: str):
        time.sleep(self._SLEEP_SEC)
        return_json = {
            'creation_datetime': REMOTE_JOB_CREATION_TIMESTAMP,
            'duration': REMOTE_JOB_DURATION,
            'failure_code': None,
            'last_intermediate_results': None,
            'msg': 'ok',
            'progress': 1.0,
            'progress_message': 'Computing phases to apply (step 2)',
            'start_time': REMOTE_JOB_START_TIMESTAMP,
            'status': 'completed',
            'status_message': None,
        }
        endpoint = self.build_endpoint(_ENDPOINT_JOB_STATUS, job_id)
        self.requests_mock.get(endpoint, json=return_json)
        return super().get_job_status(job_id)

    def get_job_results(self, job_id: str):
        time.sleep(self._SLEEP_SEC)
        endpoint = self.build_endpoint(_ENDPOINT_JOB_RESULT, job_id)
        return_json = {
            'results': json.dumps(
                {'results': serialize(REMOTE_JOB_RESULTS), 'physical_perf': 1}
            )
        }
        self.requests_mock.get(endpoint, json=return_json)
        return super().get_job_results(job_id)
