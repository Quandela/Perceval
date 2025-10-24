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
"""rpc handler module"""
from __future__ import annotations

import traceback
from urllib.parse import quote_plus

import requests
from perceval.utils.logging import get_logger, channel

_ENDPOINT_PLATFORM_DETAILS = '/api/platform/'
_ENDPOINT_PLATFORM_DETAILS_NEW = '/api/platforms/'
_PROCESSOR_SECTION = '/processor'
_ENDPOINT_JOB_CREATE = '/api/job'
_ENDPOINT_JOB_STATUS = '/api/job/status/'
_ENDPOINT_JOB_CANCEL = '/api/job/cancel/'
_ENDPOINT_JOB_RERUN = '/api/job/rerun/'
_ENDPOINT_JOB_RESULT = '/api/job/result/'
_ENDPOINT_JOB_AVAILABILITY = '/api/jobs/availability/'

_JOB_ID_KEY = 'job_id'


class RPCHandler:
    """Remote Call Procedure Handler

    A class to call the web API

    :param name: name of the target platform
    :param url: API URL to call
    :param token: token used for authentication
    :param proxies: dictionary mapping protocol to the URL of the proxy
    """

    def __init__(self, name, url, token, proxies = None):
        self.name = name
        self.url = url
        self.proxies = proxies
        self.token = token or dict()
        self.headers = {'Authorization': f'Bearer {token}'}
        self.request_timeout = 10  # default timeout

    def build_endpoint(self, endpoint: str, *args: str):
        """build the default endpoint url

        :param endpoint: first part of the endpoint
        :return: the full endpoint url from the args
        """
        endpath = ''
        if len(args) > 0:
            endpath = f"/{'/'.join(str(x).strip('/') for x in args)}"
        return f'{self.url}/{endpoint.strip("/")}{endpath}'

    def get_request(self, endpoint: str) -> None | dict:
        # requests may throw an IO Exception, let the user deal with it
        try:
            resp = requests.get(endpoint, headers=self.headers, timeout=self.request_timeout, proxies=self.proxies)
            resp.raise_for_status()
        except Exception as e:
            error_info = ''.join(traceback.format_stack()[:-1])
            get_logger().debug(error_info, channel.general)
            raise e

        try:
            return resp.json()
        except Exception as e:
            error_info = ''.join(traceback.format_stack()[:-1])
            get_logger().debug(error_info, channel.general)
            raise requests.HTTPError(f"Could not read json response from url: {endpoint}. \n{e}")

    def post_request(self, endpoint: str, payload: dict | None, with_json_response: bool) -> None | dict:
        # requests may throw an IO Exception, let the user deal with it
        try:
            request = requests.post(endpoint, headers=self.headers, json=payload, timeout=self.request_timeout, proxies=self.proxies)
        except Exception as e:
            error_info = ''.join(traceback.format_stack()[:-1])
            get_logger().debug(error_info, channel.general)
            raise e

        json_res = None
        if with_json_response:
            try:
                json_res = request.json()
            except Exception as e:
                json_res = {'error': f'{e}'}

        if request.status_code != 200:
            error_info = ''.join(traceback.format_stack())
            get_logger().debug(error_info, channel.general)
            raise requests.HTTPError(f"Url: {endpoint} answered with status code {request.status_code}.")

        return json_res

    def fetch_platform_details(self):
        """fetch platform details and settings"""
        quote_name = quote_plus(self.name)

        # first, try the new endpoint
        endpoint = self.build_endpoint(_ENDPOINT_PLATFORM_DETAILS_NEW, quote_name, _PROCESSOR_SECTION)
        try:
            response = self.get_request(endpoint)
        except requests.HTTPError:
            # try the old endpoint
            endpoint = self.build_endpoint(_ENDPOINT_PLATFORM_DETAILS, quote_name)
            response =  self.get_request(endpoint)
        return response

    def create_job(self, payload) -> str:
        """create a job

        :param payload: the payload to send
        :raises HTTPError: when the API don't accept the payload
        :return: job id
        """
        endpoint = self.build_endpoint(_ENDPOINT_JOB_CREATE)
        json_res = self.post_request(endpoint, payload, True)
        assert _JOB_ID_KEY in json_res, f'Missing {_JOB_ID_KEY} field in create_job response'
        return json_res[_JOB_ID_KEY]

    def cancel_job(self, job_id: str) -> None:
        """cancel a job

        :param job_id: id of the job to cancel
        """
        endpoint = self.build_endpoint(_ENDPOINT_JOB_CANCEL, job_id)
        self.post_request(endpoint, None, False)

    def rerun_job(self, job_id: str) -> str:
        """Rerun a job as a another freshly created job

        :param job_id: job id to rerun
        :return: id of the new job
        """
        endpoint = self.build_endpoint(_ENDPOINT_JOB_RERUN, job_id)
        json_res = self.post_request(endpoint, None, True)
        assert _JOB_ID_KEY in json_res, f'Missing {_JOB_ID_KEY} field in rerun_job response'
        return json_res[_JOB_ID_KEY]

    def get_job_status(self, job_id: str) -> dict:
        """get the status of a job

        :param job_id: if of the job
        :return: status of the job
        """
        endpoint = self.build_endpoint(_ENDPOINT_JOB_STATUS, job_id)
        return self.get_request(endpoint)

    def get_job_results(self, job_id: str) -> dict:
        """get job results

        :param job_id: id of the job
        :return: results of the job
        """
        endpoint = self.build_endpoint(_ENDPOINT_JOB_RESULT, job_id)
        return self.get_request(endpoint)

    def get_job_availability(self) -> dict:
        """get job availability
        :return: The availability of the token
        """
        endpoint = self.build_endpoint(_ENDPOINT_JOB_AVAILABILITY)
        return self.get_request(endpoint)
