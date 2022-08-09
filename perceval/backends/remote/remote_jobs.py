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

from abc import ABC
from uuid import UUID, uuid4
from .credentials import RemoteCredentials
import requests

JOB_STATUS_ENDPOINT = '/job/status/'
JOB_RESULT_ENDPOINT = '/job/result/'


class Job(ABC):
    def __init__(self, uuid: UUID | str = None, credentials: RemoteCredentials = None):
        if uuid is None:
            self.id = uuid4()
        elif isinstance(uuid, str):
            self.id = UUID(uuid)
        else:
            self.id = uuid

        self.__credentials = credentials

    def set_credentials(self, credentials: RemoteCredentials):
        if credentials is None:
            raise TypeError

        self.__credentials = credentials

    def get_status(self):
        if self.__credentials is None:
            raise TypeError

        try:
            endpoint = self.__credentials.build_endpoint(JOB_STATUS_ENDPOINT) + str(self.id)
            headers = self.__credentials.http_headers()
            res = requests.get(endpoint, headers=headers)
            json = res.json()
            return json['status'], json['msg']
        except Exception as e:
            return 'error', str(e)

    def is_completed(self):
        status = self.get_status()
        if status == 'completed':
            return True

        return False

    def result(self):
        return "ok"
