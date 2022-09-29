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
import json

from .pcvl_service import KEY_BACKEND_NAME, KEY_COUNT, \
    KEY_CIRCUIT, KEY_UNITARY, KEY_INPUT_STATE, PcvlService, check_key

from perceval import BasicState, ACircuit
from perceval.serialization import jsonstring_to_bytes, deserialize_circuit, \
    deserialize_matrix, deserialize_state, serialize

from .Q4R import Q4R, default_platform
from typing import Callable


def get_circuit_or_unitary(payload):
    cu = None
    if KEY_CIRCUIT in payload:
        cu_bytes = jsonstring_to_bytes(payload[KEY_CIRCUIT])
        cu = deserialize_circuit(cu_bytes)
    elif KEY_UNITARY in payload:
        cu_bytes = jsonstring_to_bytes(payload[KEY_UNITARY])
        cu = deserialize_matrix(cu_bytes)
    return cu


class PcvlQ4rService(PcvlService):
    default_parameters = {
        KEY_BACKEND_NAME: default_platform,
        "phi": None,
        "theta": None,
        "M": 1,
        "g2": 0,
        "eta": 1,
        "parameter_imprecision": 0
    }

    default_q4r = Q4R(default_platform)

    def __init__(self):
        self.__commands = {
            "sample_count": self.sample_count,
        }

    def get_available_commands(self):
        return self.__commands.keys()

    def get_command(self, command_name):
        return self.__commands[command_name]

    def init_computation(self, payload: dict):
        r"""
        Initiate the Q4R_plugin computer for simulation (must be changed for QPU, only phi and verifications are useful)
        """
        params = self.default_parameters | payload  # replace default parameters by existing ones

        self.q = Q4R(params[KEY_BACKEND_NAME],
                     phi=params["phi"],
                     theta=params["theta"],
                     M=params["M"],
                     g2=params["g2"],
                     eta=params["eta"],
                     q_noise=params["parameter_imprecision"]
                     )

        # Can return None if no circuit is defined; if None, phi and theta values will be used
        cu = get_circuit_or_unitary(payload)
        assert self.q.check_circuit_compatibility(cu), "circuit is not compatible"

        if isinstance(cu, ACircuit):
            self.q.input_circuit = cu

        check_key(payload, KEY_INPUT_STATE)
        self.input_state = deserialize_state(payload[KEY_INPUT_STATE])

        assert self.input_state in [BasicState([1, 0, 0, 0]), BasicState([1, 0, 1, 0])],\
            "Only |1,0,0,0> and |1,0,1,0> input states can be computed"
    """
    def sample(self, payload: dict):
        self.init_computation(payload)
        result = self.q.sample(self.input_state)
        return serialize(result)

    def samples(self, payload: dict, progress_cb: Callable):
        self.init_computation(payload)
        check_key(payload, KEY_COUNT)
        result = self.q.samples(self.input_state, payload[KEY_COUNT])
        return str([serialize(x) for x in result])"""

    def sample_count(self, payload: dict, progress_cb: Callable):
        self.init_computation(payload)
        check_key(payload, KEY_COUNT)
        result = self.q.sample_count(self.input_state, payload[KEY_COUNT])
        serialized_result = {serialize(key): count for key, count in result.items()}
        return json.dumps(serialized_result)  # Maybe some transformation is needed here for states
