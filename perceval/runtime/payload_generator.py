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
from typing import Any

from perceval.components import Experiment
from perceval.serialization import serialize
from perceval.utils import PMetadata, NoiseModel, ContextManager
from perceval.utils.constants import KEY_COMMAND, KEY_PARAMETERS, KEY_EXPERIMENT, KEY_VERSION, KEY_PROCESS_ID, \
    KEY_PAYLOAD, KEY_PLATFORM_NAME, KEY_COMPUTATION, KEY_MITIGATIONS, KEY_NOISE

from .computation import Computation
from .computation_iterator import ComputationIterator
from .error_mitigation import AbstractMitigation
from .abstract_computer import AbstractComputer

__process_id__ = uuid.uuid4()



class PayloadGenerator:

    @staticmethod
    def generate_payload(command: str,
                         experiment: Experiment = None,
                         params: dict[str, Any] = None,
                         platform_name: str = None,
                         **kwargs
                         ) -> dict[str, Any]:
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

        payload = {
            KEY_COMMAND: command,
            **kwargs
        }

        if params:
            payload[KEY_PARAMETERS] = params
        payload[KEY_EXPERIMENT] = serialize(experiment)

        global_kwargs = {KEY_PLATFORM_NAME: platform_name} if platform_name else None
        return PayloadGenerator.generate_global_data(payload, global_kwargs)

    @staticmethod
    def generate_global_data(payload: dict, kwargs: dict = None) -> dict:
        r"""
        Generate a simple payload containing the experiment, with the following template:
        {
            'pcvl_version': str
            'process_id': str
            'payload': payload
            **kwargs
        }

        :param payload: The payload to insert
        :param kwargs: other arguments to insert

        Other parameters can be added to the payload via **kwargs.
        """
        global_data = {
            KEY_VERSION: PMetadata.short_version(),
            KEY_PROCESS_ID: str(__process_id__),
            KEY_PAYLOAD: payload,
        }

        if kwargs is not None:
            for key, value in kwargs.items():
                global_data[key] = value
        return global_data

    @staticmethod
    def from_computation(computation: Computation | ComputationIterator,
                         mitigations: list[AbstractMitigation] = None,
                         parameters: dict[str, Any] = None,
                         noise: NoiseModel = None):
        payload: dict = {KEY_COMPUTATION: computation}
        if mitigations is not None:
            payload[KEY_MITIGATIONS] = mitigations
        if parameters is not None and len(parameters):
            payload[KEY_PARAMETERS] = parameters
        if noise is not None:
            payload[KEY_NOISE] = noise
        return payload

    @staticmethod
    def get_computation(payload: dict) -> Computation:
        if KEY_COMPUTATION not in payload:
            raise ValueError(f"Missing key in the payload: {KEY_COMPUTATION}")
        return payload[KEY_COMPUTATION]

    @staticmethod
    def read_configuration_from_payload(payload: dict) \
            -> tuple[list[AbstractMitigation] | None, NoiseModel | None, dict[str, Any] | None]:
        mitigations = payload.get(KEY_MITIGATIONS)
        noise = payload.get(KEY_NOISE)
        parameters = payload.get(KEY_PARAMETERS)
        return mitigations, noise, parameters

    @staticmethod
    def payload_applier(computer: AbstractComputer, payload: dict) -> ContextManager:
        """
        :param computer: The computer to configure from the payload
        :param payload: A payload, such as one generated by PayloadGenerator.from_computation()
        :return: A ContextManager that applies the parameters inside the payload to the computer (noise, mitigations, etc.)
          at enter and reset the parameters to the previous values at exit
        """
        mitigations, noise, parameters = PayloadGenerator.read_configuration_from_payload(payload)
        return computer.apply_configuration(mitigations, noise, parameters)
