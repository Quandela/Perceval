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
import uuid
import json
import pytest
import requests

import responses

from _mock_rpc_handler import RPCHandlerResponsesBuilder, DEFAULT_PLATFORM_INFO

from perceval.runtime.rpc_handler import (
    RPCHandler,
    _ENDPOINT_JOB_CANCEL,
    _ENDPOINT_JOB_CREATE,
    _ENDPOINT_JOB_RESULT,
    _ENDPOINT_JOB_STATUS,
    _ENDPOINT_PLATFORM_DETAILS,
    _ENDPOINT_JOB_RERUN,
    quote_plus
)
from perceval.runtime.job_status import RunningStatus

TOKEN = "test_token"
PLATFORM_NAME = "sim:test"
URL = "https://test"
JOB_PAYLOAD: dict[str, str] = {"key": "value"}


def test_create_job():
    rpc_handler = RPCHandler(PLATFORM_NAME, URL, TOKEN)
    RPCHandlerResponsesBuilder(rpc_handler)
    job_id = rpc_handler.create_job(JOB_PAYLOAD)

    assert str(uuid.UUID(job_id, version=4)) == job_id

    assert len(responses.calls) == 1
    create_job_request = responses.calls[0].request
    assert create_job_request.url == URL + _ENDPOINT_JOB_CREATE
    assert create_job_request.method == "POST"
    assert json.loads(create_job_request.body) == JOB_PAYLOAD
    assert create_job_request.headers['Authorization'] == f'Bearer {TOKEN}'

    RPCHandlerResponsesBuilder(rpc_handler, default_job_status=None)
    with pytest.raises(requests.exceptions.HTTPError):
        rpc_handler.create_job(JOB_PAYLOAD)


def test_get_job_infos():
    rpc_handler = RPCHandler(PLATFORM_NAME, URL, TOKEN)
    rpc_handler_responses = RPCHandlerResponsesBuilder(rpc_handler, default_job_status=RunningStatus.SUCCESS)
    job_id = rpc_handler.create_job(JOB_PAYLOAD)

    # Status
    status = rpc_handler.get_job_status(job_id)
    supposed_status = rpc_handler_responses.get_job_status_response_body_from_job_status(RunningStatus.SUCCESS)
    assert len(status) == len(supposed_status)
    for key in status:
        if isinstance(status[key], float):
            assert status[key] == pytest.approx(supposed_status[key])
        else:
            status[key] == supposed_status[key]
    assert status['status'] == "completed"
    assert int(status['duration']) >= 0
    assert float(status['progress']) == 1.0

    assert len(responses.calls) == 2
    job_status_request = responses.calls[1].request
    assert job_status_request.url == URL + _ENDPOINT_JOB_STATUS + job_id
    assert job_status_request.method == "GET"
    assert job_status_request.body == None
    assert job_status_request.headers['Authorization'] == f'Bearer {TOKEN}'

    # Results
    results = rpc_handler.get_job_results(job_id)
    assert results == rpc_handler_responses.get_job_result_response_body_from_job_status(RunningStatus.SUCCESS)
    assert "results" in results
    assert "results" is not None

    assert len(responses.calls) == 3
    job_results_request = responses.calls[2].request
    assert job_results_request.url == URL + _ENDPOINT_JOB_RESULT + job_id
    assert job_results_request.method == "GET"
    assert job_results_request.body == None
    assert job_results_request.headers['Authorization'] == f'Bearer {TOKEN}'

    with pytest.raises(requests.exceptions.HTTPError):
        rpc_handler.get_job_status(str(uuid.uuid4()))

    with pytest.raises(requests.exceptions.HTTPError):
        rpc_handler.get_job_results(str(uuid.uuid4()))


def test_cancel_rerun_job():
    rpc_handler = RPCHandler(PLATFORM_NAME, URL, TOKEN)

    RPCHandlerResponsesBuilder(rpc_handler)
    with pytest.raises(requests.exceptions.HTTPError):
        rpc_handler.cancel_job(str(uuid.uuid4()))
    with pytest.raises(requests.exceptions.HTTPError):
        rpc_handler.rerun_job(str(uuid.uuid4()))

    # cancel
    for status in [RunningStatus.RUNNING, RunningStatus.WAITING]:
        RPCHandlerResponsesBuilder(rpc_handler, default_job_status=status)

        job_id = rpc_handler.create_job(JOB_PAYLOAD)

        rpc_handler.cancel_job(job_id)

        assert len(responses.calls) == 2

        cancel_job_request = responses.calls[1].request
        assert cancel_job_request.url == URL + _ENDPOINT_JOB_CANCEL + job_id
        assert cancel_job_request.method == "POST"
        assert cancel_job_request.body == None
        assert cancel_job_request.headers['Authorization'] == f'Bearer {TOKEN}'

    for status in [RunningStatus.SUCCESS, RunningStatus.ERROR, RunningStatus.CANCELED]:
        RPCHandlerResponsesBuilder(rpc_handler, default_job_status=status)
        job_id = rpc_handler.create_job(JOB_PAYLOAD)
        with pytest.raises(requests.exceptions.HTTPError):
            rpc_handler.cancel_job(job_id)

    # rerun
    for status in [RunningStatus.ERROR, RunningStatus.CANCELED]:
        RPCHandlerResponsesBuilder(rpc_handler, default_job_status=status)

        job_id = rpc_handler.create_job(JOB_PAYLOAD)

        new_job_id = rpc_handler.rerun_job(job_id)
        assert job_id != new_job_id
        assert str(uuid.UUID(new_job_id, version=4)) == new_job_id

        assert len(responses.calls) == 2

        rerun_job_request = responses.calls[1].request
        assert rerun_job_request.url == URL + _ENDPOINT_JOB_RERUN + job_id
        assert rerun_job_request.method == "POST"
        assert rerun_job_request.body == None
        assert rerun_job_request.headers['Authorization'] == f'Bearer {TOKEN}'

    for status in [RunningStatus.SUCCESS, RunningStatus.WAITING, RunningStatus.RUNNING]:
        RPCHandlerResponsesBuilder(rpc_handler, default_job_status=status)
        job_id = rpc_handler.create_job(JOB_PAYLOAD)
        with pytest.raises(requests.exceptions.HTTPError):
            rpc_handler.rerun_job(job_id)


def test_get_platform_details():
    rpc_handler = RPCHandler(PLATFORM_NAME, URL, TOKEN)

    platform_info = DEFAULT_PLATFORM_INFO
    platform_info['name'] = PLATFORM_NAME

    RPCHandlerResponsesBuilder(rpc_handler, platform_info=platform_info)

    assert platform_info == rpc_handler.fetch_platform_details()
