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

import logging
import time
from json import JSONDecodeError

import requests

from ..template import AbstractBackend
from .credentials import RemoteCredentials
from .remote_jobs import Job

from perceval.serialization import serialize


SAMPLE_ENDPOINT = '/api/1.0/backend/sample'


class RemoteBackend(AbstractBackend):
    def __init__(self, name, credentials: RemoteCredentials):
        self.name = name
        self.__credentials = credentials

    def sample(self, input_state):
        job = self.async_sample(input_state)
        while True:
            if not job.is_completed():
                time.sleep(1)
            else:
                return job.result()

    def async_sample(self, input_state):
        body = {
            'backend_name': self.name,
            'input_states': serialize(input_state)
        }

        endpoint = self.__credentials.build_endpoint(SAMPLE_ENDPOINT)
        request = requests.post(endpoint,
                                headers=self.__credentials.http_headers(), data=body)
        request.raise_for_status()

        job = None
        try:
            json = request.json()
            job = Job(json['job_id'], self.__credentials)
        except JSONDecodeError as ex:
            logging.error(f"Could not load response :{ex.msg}")

        return job
