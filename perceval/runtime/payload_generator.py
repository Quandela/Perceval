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

from perceval.components import Experiment
from perceval.serialization import serialize
from perceval.utils import PMetadata

__process_id__ = uuid.uuid4()



class PayloadGenerator:

    @staticmethod
    def generate_payload(command: str,
                         experiment: Experiment = None,
                         params: dict[str, any] = None,
                         platform_name: str = "",
                         **kwargs
                         ) -> dict[str, any]:
        r"""
        Generate a simple payload containing the experiment, with the following template:
        {
            'pcvl_version': ...
            'process_id': ...
            'platform_name': ...
            'payload': {
                'command': ...
                'kwarg1': ...
                'kwarg2': ...
                ...
                'parameters': ...
                'experiment': ...
            }
        }

        :param command: name of the method used
        :param experiment: (optional) Perceval experiment, by default an empty Experiment will be generated
        :param params: (optional) parameters to be listed in the 'parameters' section of the payload
        :param platform_name: (optional) name of the platform used

        Other parameters can be added to the payload via **kwargs.
        """

        if experiment is None:
            experiment = Experiment()

        j = {
            'pcvl_version': PMetadata.short_version(),
            'process_id': str(__process_id__)
        }
        if platform_name:
            j['platform_name'] = platform_name

        payload = {
            'command': command,
            **kwargs
        }

        if params:
            payload['parameters'] = params

        payload['experiment'] = serialize(experiment)
        j['payload'] = payload

        return j
