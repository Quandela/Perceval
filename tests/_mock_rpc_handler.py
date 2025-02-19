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
from __future__ import annotations

import uuid
import json
import re
import datetime
from enum import Enum

import responses

from perceval.runtime.rpc_handler import (
    RPCHandler,
    _ENDPOINT_JOB_CANCEL,
    _ENDPOINT_JOB_CREATE,
    _ENDPOINT_JOB_RESULT,
    _ENDPOINT_JOB_STATUS,
    _ENDPOINT_PLATFORM_DETAILS,
    _ENDPOINT_JOB_RERUN,
    _JOB_ID_KEY,
    quote_plus
)
from perceval.runtime.job_status import RunningStatus, JobStatus

_TIMESTAMP = datetime.datetime.now().timestamp()

DEFAULT_PLATFORM_INFO = {
    'id': str(uuid.uuid4()),
    'name': None,
    'perfs': {},
    'specs': {
        'available_commands': ['probs'],
        'connected_input_modes': [0, 2, 4, 6, 8, 10],
        'constraints': {
            'max_mode_count': 20,
            'max_photon_count': 6,
            'min_mode_count': 1,
            'min_photon_count': 1,
        }
    },
    'status': 'available',
    'type': 'simulator',
}

UUID_REGEXP = "[a-f0-9]{8}-[a-f0-9]{4}-4[a-f0-9]{3}-[89aAbB][a-f0-9]{3}-[a-f0-9]{12}"


class CloudEndpoint(Enum):
    CreateJob = 0
    JobStatus = 1
    JobResults = 2
    CancelJob = 3
    RerunJob = 4
    PlatformDetails = 5

    @staticmethod
    def from_response(response: responses.Response) -> CloudEndpoint:
        if _ENDPOINT_JOB_STATUS in response.url:
            return CloudEndpoint.JobStatus
        if _ENDPOINT_JOB_RESULT in response.url:
            return CloudEndpoint.JobResults
        if _ENDPOINT_JOB_CANCEL in response.url:
            return CloudEndpoint.CancelJob
        if _ENDPOINT_JOB_RERUN in response.url:
            return CloudEndpoint.RerunJob
        if _ENDPOINT_JOB_CREATE in response.url:
            return CloudEndpoint.CreateJob
        if _ENDPOINT_PLATFORM_DETAILS in response.url:
            return CloudEndpoint.PlatformDetails


class RPCHandlerResponsesBuilder():

    def __init__(self,
                 rpc_handler: RPCHandler,
                 platform_info: dict = DEFAULT_PLATFORM_INFO,
                 default_job_status: RunningStatus | None = RunningStatus.SUCCESS) -> None:
        self._rpc_handler = rpc_handler
        platform_info['name'] = rpc_handler.name
        self._platform_info = platform_info
        self._job_status = default_job_status
        responses.reset()
        self._set_default_responses()

    def set_default_job_status(self, default_job_status: RunningStatus | None = RunningStatus.SUCCESS) -> None:
        self._job_status = default_job_status

    def _set_default_responses(self) -> None:
        self._set_get_platform_details_responses()
        for method, endpoint in [
            ('POST', _ENDPOINT_JOB_RERUN),
            ('POST', _ENDPOINT_JOB_CANCEL),
            ('GET', _ENDPOINT_JOB_STATUS),
            ('GET', _ENDPOINT_JOB_RESULT),
            ('GET', _ENDPOINT_PLATFORM_DETAILS)
        ]:
            responses.add(responses.Response(
                method=method,
                url=re.compile((self._rpc_handler.url + endpoint).replace('/', "\/") + UUID_REGEXP),
                status=404))

        responses.add_callback(responses.POST,
                               self._rpc_handler.url + _ENDPOINT_JOB_CREATE,
                               callback=self.create_job_callback)

    def _reset_default_responses(self) -> None:
        for method, endpoint in [
            ('POST', _ENDPOINT_JOB_RERUN),
            ('POST', _ENDPOINT_JOB_CANCEL),
            ('GET', _ENDPOINT_JOB_STATUS),
            ('GET', _ENDPOINT_JOB_RESULT),
            ('GET', _ENDPOINT_PLATFORM_DETAILS)
        ]:
            responses.remove(responses.Response(
                method=method,
                url=re.compile((self._rpc_handler.url + endpoint).replace('/', "\/") + UUID_REGEXP),
                status=404))
            responses.add(responses.Response(
                method=method,
                url=re.compile((self._rpc_handler.url + endpoint).replace('/', "\/") + UUID_REGEXP),
                status=404))

    def create_job_callback(self, _):
        if self._job_status is None:
            return (400, {"content-type": "application/json"}, "")
        job_id = str(uuid.uuid4())
        self._set_rerun_job_responses(job_id, self._job_status)
        self._set_cancel_job_responses(job_id, self._job_status)
        self._set_job_status_responses(job_id, self._job_status)
        self._set_job_result_responses(job_id, self._job_status)
        self._reset_default_responses()
        return (200, {"content-type": "application/json"}, json.dumps({_JOB_ID_KEY: job_id}))

    def _set_rerun_job_responses(self, job_id: str, status: RunningStatus = RunningStatus.SUCCESS) -> None:
        job_status = JobStatus()
        job_status.status = status
        if job_status.failed:
            responses.head
            responses.add_callback(
                responses.POST,
                self._rpc_handler.url + _ENDPOINT_JOB_RERUN + job_id,
                callback=self.create_job_callback)
        else:
            responses.add(responses.Response(
                method='POST',
                url=self._rpc_handler.url + _ENDPOINT_JOB_RERUN + job_id,
                status=400))

    def _set_cancel_job_responses(self, job_id: str, status: RunningStatus = RunningStatus.SUCCESS) -> None:
        job_status = JobStatus()
        job_status.status = status
        if job_status.running or job_status.waiting:
            responses.add(responses.Response(
                method='POST',
                url=self._rpc_handler.url + _ENDPOINT_JOB_CANCEL + job_id,
                status=200))
        else:
            responses.add(responses.Response(
                method='POST',
                url=self._rpc_handler.url + _ENDPOINT_JOB_CANCEL + job_id,
                status=400))

    def get_job_status_response_body_from_job_status(self, status: RunningStatus):
        response_body = {
            'duration': None,
            'progress': None,
            'status': RunningStatus.to_server_response(status),
            'creation_date': _TIMESTAMP,
            'start_time': None,
            'failure_code': None
        }

        if status == RunningStatus.RUNNING:
            response_body['progress'] = 0.5
            response_body['duration'] = 10
            response_body['start_time'] = response_body['creation_date'] + 1.
        elif status == RunningStatus.SUCCESS:
            response_body['progress'] = 1.0
            response_body['duration'] = 20
            response_body['start_time'] = response_body['creation_date'] + 1.
        elif status == RunningStatus.CANCELED:
            response_body['status_message'] = 'Cancel requested from web interface'

        return response_body

    def _set_job_status_responses(self, job_id: str, status: RunningStatus = RunningStatus.SUCCESS) -> None:
        responses.add(responses.Response(
            method='GET',
            url=self._rpc_handler.url + _ENDPOINT_JOB_STATUS + job_id,
            status=200,
            json=self.get_job_status_response_body_from_job_status(status)))

    def get_job_result_response_body_from_job_status(self, status: RunningStatus):
        response_body = {
            'results': None
        }
        if status == RunningStatus.SUCCESS:
            response_body['results'] = json.dumps(
                {'results': ':PCVL:BasicState:|>', 'logical_perf': 1, 'physical_perf': 0.1})
        return response_body

    def _set_job_result_responses(self, job_id: str, status: RunningStatus = RunningStatus.SUCCESS) -> None:
        responses.add(responses.Response(
            method='GET',
            url=self._rpc_handler.url + _ENDPOINT_JOB_RESULT + job_id,
            status=200,
            json=self.get_job_result_response_body_from_job_status(status)))

    def _set_get_platform_details_responses(self) -> None:
        responses.add(responses.Response(
            method='GET',
            url=self._rpc_handler.url + _ENDPOINT_PLATFORM_DETAILS + quote_plus(self._rpc_handler.name),
            status=200,
            json=self._platform_info))


def get_rpc_handler_for_tests(platform_name: str = "sim:test", url: str = "https://test", token="test_token"):
    rpc_handler = RPCHandler(platform_name, url, token)
    RPCHandlerResponsesBuilder(rpc_handler)
    return rpc_handler
