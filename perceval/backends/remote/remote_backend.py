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
from typing import Union

import requests

from ..template import AbstractBackend
from .credentials import RemoteCredentials
from .remote_jobs import Job

from perceval.serialization import serialize, bytes_to_jsonstring
from perceval.components import ACircuit
from perceval.utils import Matrix

from pkg_resources import get_distribution
pcvl_version = get_distribution("perceval-quandela").version

JOB_CREATE_ENDPOINT = '/api/job'


class RemoteBackendBuilder:
    def __init__(self, name: str, platform: str, credentials: RemoteCredentials):
        self.name = name
        self.platform = platform
        self.credentials = credentials

    def __call__(self, u, use_symbolic=None, n=None, mask=None):
        return RemoteBackend(self.name, self.platform, self.credentials, u, use_symbolic, n, mask)


class RemoteBackend(AbstractBackend):
    def __init__(self, name: str, platform: str, credentials: RemoteCredentials, cu: Union[ACircuit, Matrix], use_symbolic=None, n=None, mask=None):
        self.name = name
        self.__platform = platform
        self.__credentials = credentials
        if isinstance(cu, ACircuit):
            self.__cu_key = 'circuit'
        else:
            self.__cu_key = 'unitary'
        self.__cu_data = bytes_to_jsonstring(serialize(cu))

    def __defaults_payload(self, command):
        return {
            'platform': self.__platform,
            'command': command,
            'pcvl_version': pcvl_version
        }

    def __create_job_endpoint(self, body):
        job = None
        try:
            endpoint = self.__credentials.build_endpoint(JOB_CREATE_ENDPOINT)
            request = requests.post(endpoint,
                                    headers=self.__credentials.http_headers(),
                                    json=body)
            request.raise_for_status()

            json = request.json()
            job = Job(json['id'], self.__credentials)
        except ConnectionError as e:
            logging.error(f"Connection error: {str(e)}")

        except JSONDecodeError as ex:
            logging.error(f"Could not load response :{ex.msg}")
        return job

    def sample(self, input_state):
        job = self.async_sample(input_state)
        while True:
            if not job.is_completed():
                time.sleep(2)
            else:
                return job.result()

    def async_sample(self, input_state):
        payload = self.__defaults_payload('sample')
        payload['data'] = {
            'backend_name': self.name,
            self.__cu_key: self.__cu_data,
            'input_state': serialize(input_state)
        }

        return self.__create_job_endpoint(payload)

    def async_samples(self, input_state, count):
        payload = self.__defaults_payload('samples')
        payload['data'] = {
            'backend_name': self.name,
            self.__cu_key: self.__cu_data,
            'input_state': serialize(input_state),
            'count': count
        }

        return self.__create_job_endpoint(payload)
