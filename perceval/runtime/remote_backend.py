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

from perceval.backends import Backend
from perceval.components import ACircuit
from .rpc_handler import RPCHandler
from perceval.serialization import serialize
from perceval.utils import Matrix, BasicState, generate_sync_methods

from pkg_resources import get_distribution

pcvl_version = get_distribution("perceval-quandela").version


################################
@generate_sync_methods
class RemoteBackend(Backend):

    def __init__(self, rpc: RPCHandler, backend_name, cu: Union[ACircuit, Matrix], use_symbolic=None, n=None,
                 mask=None):
        self.name = backend_name
        self.__rpc_handler = rpc
        if isinstance(cu, ACircuit):
            self.__cu_key = 'circuit'
        else:
            self.__cu_key = 'unitary'
        self.__cu_data = serialize(cu)
        super(RemoteBackend, self).__init__(cu, use_symbolic, n, mask)

    @staticmethod
    def preferred_command() -> str:
        return 'sample_count'

    def __defaults_job_params(self, command: str):
        return {
            'platform_name': self.__rpc_handler.name,
            'job_name': command,
            'pcvl_version': pcvl_version,
            'backend_name': self.name,
            self.__cu_key: self.__cu_data
        }

    def async_samples(self, input_state, count, parameters=None):
        return self.async_execute('samples', parameters,
                                  args={'input_state': serialize(input_state),
                                        'count': count})

    def async_sample_count(self, input_state, count, parameters=None):
        return self.async_execute('sample_count', parameters,
                                  args={'input_state': serialize(input_state),
                                        'count': count})

    def async_probs(self, input_state: BasicState, parameters=None):
        return self.async_execute('probs', parameters,
                                  args={'input_state': serialize(input_state)})

    def async_execute(self, command: str, parameters=None, **args):
        job_params = self.__defaults_job_params(command)
        job_params['payload'] = {
            **args
        }
        if parameters is not None:
            job_params['payload']['parameters'] = parameters

        return self.__rpc_handler.create_job(job_params)


    def probampli_be(self, input_state, output_state, n=None):
        raise NotImplementedError

    def prob_be(self, input_state, output_state, n=None):
        raise NotImplementedError
