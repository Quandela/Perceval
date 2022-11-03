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

from perceval.components import ACircuit
from .rpc_handler import RPCHandler
from perceval.serialization import serialize
from perceval.utils import Matrix, BasicState

from pkg_resources import get_distribution

pcvl_version = get_distribution("perceval-quandela").version


################################
class RemoteBackend:

    def __init__(self, rpc: RPCHandler, cu: Union[ACircuit, Matrix]):
        self.__rpc_handler = rpc
        if isinstance(cu, ACircuit):
            self.__cu_key = 'circuit'
        else:
            self.__cu_key = 'unitary'
        self.__cu_data = serialize(cu)

    def __defaults_job_params(self, command: str):
        return {
            'platform_name': self.__rpc_handler.name,
            'job_name': command,
            'pcvl_version': pcvl_version
        }

    def async_execute(self, command: str, parameters=None, **args):
        job_params = self.__defaults_job_params(command)
        job_params['payload'] = {
            'command': command,
            self.__cu_key: self.__cu_data,
            **args
        }

        if parameters:
            job_params['payload']['parameters'] = parameters

        return self.__rpc_handler.create_job(serialize(job_params))
