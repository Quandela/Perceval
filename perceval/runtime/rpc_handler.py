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
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import urllib

import requests
from requests import HTTPError

_ENDPOINT_PLATFORM_DETAILS = '/api/platform/'
_ENDPOINT_JOB_CREATE = '/api/job'
_ENDPOINT_JOB_STATUS = '/api/job/status/'
_ENDPOINT_JOB_CANCEL = '/api/job/cancel/'
_ENDPOINT_JOB_RESULT = '/api/job/result/'


class RPCHandler:
    def __init__(self, name, url, token):
        self.name = name
        self.url = url
        self.token = token
        self.headers = {
            'Authorization': f'Bearer {token}'
        }

    def build_endpoint(self, endpoint):
        return f"{self.url}{endpoint}"

    def fetch_platform_details(self):
        endpoint = f"{self.url}{_ENDPOINT_PLATFORM_DETAILS}{urllib.parse.quote_plus(self.name)}"
        resp = requests.get(endpoint,
                            headers=self.headers)
        resp.raise_for_status()
        return resp.json()

    def create_job(self, payload):
        endpoint = f"{self.url}{_ENDPOINT_JOB_CREATE}"
        request = requests.post(endpoint,
                                headers=self.headers,
                                json=payload)

        json = {}
        try:
            jsonres = request.json()
        except:
            pass

        if request.status_code != 200:
            raise HTTPError(jsonres['error'])

        return jsonres['job_id']

    def cancel_job(self, job_id: str):
        endpoint = f"{self.url}{_ENDPOINT_JOB_CANCEL}{str(job_id)}"
        request = requests.post(endpoint,
                                headers=self.headers)
        request.raise_for_status()

    def get_job_status(self, job_id: str):
        endpoint = self.build_endpoint(_ENDPOINT_JOB_STATUS) + str(job_id)

        # requests may throw an IO Exception, let the user deal with it
        res = requests.get(endpoint, headers=self.headers)
        res.raise_for_status()
        return res.json()

    def get_job_results(self, job_id: str):
        endpoint = self.build_endpoint(_ENDPOINT_JOB_RESULT) + str(job_id)

        # requests may throw an IO Exception, let the user deal with it
        res = requests.get(endpoint, headers=self.headers)
        res.raise_for_status()
        return res.json()
