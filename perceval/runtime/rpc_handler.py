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

from urllib.parse import quote_plus

import requests
from requests import HTTPError

_ENDPOINT_PLATFORM_DETAILS = '/api/platform/'
_ENDPOINT_JOB_CREATE = '/api/job'
_ENDPOINT_JOB_STATUS = '/api/job/status/'
_ENDPOINT_JOB_CANCEL = '/api/job/cancel/'
_ENDPOINT_JOB_RERUN = '/api/job/rerun/'
_ENDPOINT_JOB_RESULT = '/api/job/result/'

_JOB_ID_KEY = 'job_id'


class RPCHandler:
    """Remote Call Procedure Handler

    A class to call the API

    """

    def __init__(self, name, url, token):
        """Remote Call Procedure Handler

        :param name: name of the plateform
        :param url: api URL to call
        :param token: token used for identification
        """
        self.name = name
        self.url = url
        self.token = token
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

    def fetch_platform_details(self):
        """fetch platform details and settings"""
        quote_name = quote_plus(self.name)
        endpoint = self.build_endpoint(_ENDPOINT_PLATFORM_DETAILS, quote_name)
        resp = requests.get(endpoint, headers=self.headers, timeout=self.request_timeout)
        resp.raise_for_status()
        return resp.json()

    def create_job(self, payload):
        """create a job

        :param payload: the payload to send
        :raises HTTPError: when the API don't accept the payload
        :return: job id
        """
        endpoint = self.build_endpoint(_ENDPOINT_JOB_CREATE)
        request = requests.post(endpoint, headers=self.headers, json=payload, timeout=self.request_timeout)
        try:
            json_res = request.json()
        except Exception as e:
            json_res = {'error': f'{e}'}

        if request.status_code != 200:
            raise HTTPError(json_res.get('error', 'Unspecified error'))

        return json_res[_JOB_ID_KEY]

    def cancel_job(self, job_id: str):
        """cancel a job

        :param job_id: id of the job
        """
        endpoint = self.build_endpoint(_ENDPOINT_JOB_CANCEL, job_id)
        req = requests.post(endpoint, headers=self.headers, timeout=self.request_timeout)
        req.raise_for_status()

    def rerun_job(self, job_id: str) -> str:
        """Rerun a job

        :param job_id: job id to rerun
        :return: new job id
        """
        endpoint = self.build_endpoint(_ENDPOINT_JOB_RERUN, job_id)
        req = requests.post(endpoint, headers=self.headers, timeout=self.request_timeout)
        req.raise_for_status()
        assert _JOB_ID_KEY in req.json(), f'Missing {_JOB_ID_KEY} field in rerun response'
        return req.json()[_JOB_ID_KEY]

    def get_job_status(self, job_id: str):
        """get the status of a job

        :param job_id: if of the job
        :return: status of the job
        """
        endpoint = self.build_endpoint(_ENDPOINT_JOB_STATUS, job_id)

        # requests may throw an IO Exception, let the user deal with it
        res = requests.get(endpoint, headers=self.headers, timeout=self.request_timeout)
        res.raise_for_status()
        return res.json()

    def get_job_results(self, job_id: str):
        """get job results

        :param job_id: id of the job
        :return: results of the job
        """
        endpoint = self.build_endpoint(_ENDPOINT_JOB_RESULT, job_id)

        # requests may throw an IO Exception, let the user deal with it
        res = requests.get(endpoint, headers=self.headers, timeout=self.request_timeout)
        res.raise_for_status()
        return res.json()
