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
import json
from perceval.utils import BSDistribution, BasicState
from perceval.serialization import serialize


REMOTE_JOB_NAME = "a remote job"
REMOTE_JOB_DURATION = 5
REMOTE_JOB_CREATION_TIMESTAMP = 1687883254.77622
REMOTE_JOB_START_TIMESTAMP = 1687883263.280909
REMOTE_JOB_RESULTS = BSDistribution({
    BasicState([1, 0, 0, 0]): 0.200266,
    BasicState([0, 1, 0, 0]): 0.09734,
    BasicState([0, 0, 1, 0]): 0.089365,
    BasicState([0, 0, 0, 1]): 0.223731,
    BasicState([1, 0, 1, 0]): 0.308951
})


class MockRPCHandler:
    _ARBITRARY_JOB_ID = "ebb1f8ec-0125-474f-9ffc-5178afef4d1a"
    _SLEEP_SEC = 0.2
    name = 'mocked:platform'

    def fetch_platform_details(self):
        time.sleep(self._SLEEP_SEC)
        return {
            'created_date': 'Mon, 31 Oct 2022 16:54:45 GMT',
            'description': 'Mocked Simulator',
            'id': 'e576e49c-7b1a-470b-5910-c04e406d40f6',
            'jobs': 6687,
            'name': self.name,
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
        }

    def create_job(self, payload):
        time.sleep(self._SLEEP_SEC)
        return self._ARBITRARY_JOB_ID

    def cancel_job(self, job_id: str):
        time.sleep(self._SLEEP_SEC)

    def get_job_status(self, job_id: str):
        time.sleep(self._SLEEP_SEC)
        return {
            "creation_datetime": REMOTE_JOB_CREATION_TIMESTAMP,
            "duration": REMOTE_JOB_DURATION,
            "failure_code": None,
            "last_intermediate_results": None,
            "msg": "ok",
            "name": REMOTE_JOB_NAME,
            "progress": 1.0,
            "progress_message": "Computing phases to apply (step 2)",
            "start_time": REMOTE_JOB_START_TIMESTAMP,
            "status": "completed",
            "status_message": None
        }

    def get_job_results(self, job_id: str):
        time.sleep(self._SLEEP_SEC)
        return {'results': json.dumps({
            'results': serialize(REMOTE_JOB_RESULTS),
            'physical_perf': 1
        })}
