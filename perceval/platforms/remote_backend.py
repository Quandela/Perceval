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

from typing import Union
import logging
import requests
import time
from json import JSONDecodeError

from perceval.backends import Backend
from perceval.components import ACircuit
from perceval.serialization import serialize, bytes_to_jsonstring, deserialize_state, deserialize_state_list, \
    deserialize_float
from perceval.utils import Matrix, BasicState

from pkg_resources import get_distribution

pcvl_version = get_distribution("perceval-quandela").version
JOB_CREATE_ENDPOINT = '/api/job'

################################
# Sync methods Factory
def _sync_wrapper(cls, func):
    async_func = getattr(cls, func)

    def await_job(*args):
        job = async_func(*args)
        while True:
            if not job.is_completed():
                time.sleep(3)
            else:
                return job.get_results()

    return await_job


def generate_sync_methods(cls):
    for method in dir(cls):
        if method.startswith('async_'):
            sync_name = method.removeprefix('async_')
            setattr(cls, sync_name, _sync_wrapper(cls, method))
    return cls


################################
@generate_sync_methods
class RemoteBackend(Backend):
    def __init__(self, platform, cu: Union[ACircuit, Matrix], use_symbolic=None, n=None, mask=None):
        self.name = platform.name
        self.__platform = platform
        if isinstance(cu, ACircuit):
            self.__cu_key = 'circuit'
        else:
            self.__cu_key = 'unitary'
        self.__cu_data = bytes_to_jsonstring(serialize(cu))
        super(RemoteBackend, self).__init__(cu, use_symbolic, n, mask)

    def __defaults_payload(self, command: str):
        return {
            'platform_id': self.name,
            'job_name': command,
            'pcvl_version': pcvl_version
        }

    def __request_job_create(self, body):
        job = None
        try:
            endpoint = self.__platform.build_endpoint(JOB_CREATE_ENDPOINT)
            request = requests.post(endpoint,
                                    headers=self.__platform.get_http_headers(),
                                    json=body)
            request.raise_for_status()

            json = request.json()
            # job = Job(json['job_id'], self.__credentials)
        except ConnectionError as e:
            logging.error(f"Connection error: {str(e)}")

        except JSONDecodeError as ex:
            logging.error(f"Could not load response :{ex.msg}")
        return job

    def prob_be(self, input_state, output_state, n=None):
        raise NotImplementedError

    def async_sample(self, input_state):
        payload = self.__defaults_payload('sample')
        payload['payload'] = {
            'backend_name': self.name,
            self.__cu_key: self.__cu_data,
            'input_state': serialize(input_state)
        }

        job = self.__request_job_create(payload)
        job.set_deserializer(deserialize_state)

        return job

    def async_samples(self, input_state, count):
        payload = self.__defaults_payload('samples')
        payload['payload'] = {
            'backend_name': self.name,
            self.__cu_key: self.__cu_data,
            'input_state': serialize(input_state),
            'count': count
        }

        job = self.__request_job_create(payload)
        job.set_deserializer(deserialize_state_list)
        return job

    def async_prob(self,
                   input_state: BasicState,
                   output_state: BasicState,
                   n: int = None,
                   skip_compile: bool = False):
        payload = self.__defaults_payload('prob')
        payload['payload'] = {
            'backend_name': self.name,
            self.__cu_key: self.__cu_data,
            'input_state': serialize(input_state),
            'output_state': serialize(output_state),
            'n': n,
            'skip_compile': skip_compile
        }

        job = self.__request_job_create(payload)
        job.set_deserializer(deserialize_float)
        return job
