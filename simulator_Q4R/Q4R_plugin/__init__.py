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
from perceval.serialization import serialize
from .q4r_pcvl import PcvlQ4rService
from typing import Callable
from .Q4R import Q4R, default_platform
from perceval import BasicState
from perceval.serialization import bytes_to_jsonstring


q = Q4R(default_platform)


class Plugin:
    qrng = PcvlQ4rService()

    def register(self):
        return {
            "pcvl_version": self.qrng.version,
            "available_commands": list(self.get_available_commands()),
            "specific_circuit": bytes_to_jsonstring(serialize(q.input_circuit)),
            "accepted_input_states": [serialize(BasicState([1, 0, 0, 0])), serialize(BasicState([1, 0, 1, 0]))]
        }

    def can_handle_command(self, command: str):
        return command in self.qrng.get_available_commands()

    def get_available_commands(self):
        return self.qrng.get_available_commands()

    def exec_command(self, command: str, data: dict, progress_cb: Callable):
        pcvl_cmd = self.qrng.get_command(command)
        result = pcvl_cmd(data, progress_cb)
        return result
