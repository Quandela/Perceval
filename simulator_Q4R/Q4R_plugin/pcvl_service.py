from typing import Callable

import perceval as pcvl
from perceval.algorithm import Sampler
from perceval.serialization import deserialize_state, deserialize_matrix, deserialize_circuit, jsonstring_to_bytes, \
    serialize

KEY_BACKEND_NAME = 'backend_name'
KEY_CIRCUIT = 'circuit'
KEY_UNITARY = 'unitary'
KEY_INPUT_STATE = 'input_state'
KEY_OUTPUT_STATE = 'output_state'
KEY_COUNT = 'count'
KEY_N = 'n'
KEY_SKIP_COMPILE = 'skip_compile'


def check_key(var, key):
    if key not in var:
        raise TypeError(f"Missing key : {key}")


def _get_circuit_or_unitary(payload):
    cu = None
    if KEY_CIRCUIT in payload:
        cu_bytes = jsonstring_to_bytes(payload[KEY_CIRCUIT])
        cu = deserialize_circuit(cu_bytes)
    elif KEY_UNITARY in payload:
        cu_bytes = jsonstring_to_bytes(payload[KEY_UNITARY])
        cu = deserialize_matrix(cu_bytes)
    else:
        raise ValueError("Missing Circuit or Unitary")
    return cu


class PcvlService:
    version = pcvl.__version__

    def __init__(self):
        self.__commands = {
            "sample": self.sample,
            "samples": self.samples
        }

    def get_available_commands(self):
        return self.__commands.keys()

    def get_command(self, command_name):
        return self.__commands[command_name]

    @staticmethod
    def list_backends():
        return pcvl.BACKEND_LIST.keys()

    @staticmethod
    def sample(payload: dict):
        check_key(payload, KEY_BACKEND_NAME)
        check_key(payload, KEY_INPUT_STATE)

        input_state = deserialize_state(payload[KEY_INPUT_STATE])

        cu = _get_circuit_or_unitary(payload)
        platform = pcvl.get_platform(payload[KEY_BACKEND_NAME])
        sampler = Sampler(platform, cu)

        result = sampler.sample(input_state)
        return serialize(result)

    @staticmethod
    def samples(payload: dict, progress_cb: Callable):
        check_key(payload, KEY_BACKEND_NAME)
        check_key(payload, KEY_INPUT_STATE)
        check_key(payload, KEY_COUNT)

        input_state = deserialize_state(payload[KEY_INPUT_STATE])

        cu = _get_circuit_or_unitary(payload)
        platform = pcvl.get_platform(payload[KEY_BACKEND_NAME])
        sampler = Sampler(platform, cu)

        result = sampler.samples(input_state, payload[KEY_COUNT])
        return str([serialize(x) for x in result])

    # def prob(self, payload: dict):
    #     check_key(payload, KEY_BACKEND_NAME)
    #     check_key(payload, KEY_INPUT_STATE)
    #     check_key(payload, KEY_OUTPUT_STATE)
    #     check_key(payload, KEY_N)
    #     check_key(payload, KEY_SKIP_COMPILE)
    #
    #     input_state = deserialize_state(payload[KEY_INPUT_STATE])
    #     output_state = deserialize_state(payload[KEY_OUTPUT_STATE])
    #
    #     cu = _get_circuit_or_unitary(payload)
    #     pcvl_backend = self.__build_backend(payload[KEY_BACKEND_NAME], cu)
    #
    #     result = pcvl_backend.prob(input_state, output_state, payload[KEY_N], payload[KEY_SKIP_COMPILE])
    #     return str(result)
