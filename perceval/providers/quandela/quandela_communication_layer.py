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
from requests import HTTPError

from perceval.runtime import PayloadGenerator
from providers.quandela.rpc_handler import RPCHandler
from perceval.runtime.communication_layer import RPCBasedCommunicationLayer, RemoteId

from perceval.utils.logging import get_logger, channel
from perceval.utils.constants import KEY_COMMAND, KEY_MAX_SHOTS


class QuandelaCommunicationLayer(RPCBasedCommunicationLayer):

    def __init__(self, name: str, token: str, url: str, proxies: dict[str, str] = None):
        super().__init__(RPCHandler(name, url, token, proxies))
        get_logger().info(f"Connected to Cloud platform {name}", channel.general)

    def send(self, payload: dict) -> RemoteId:
        computation = PayloadGenerator.get_computation(payload)

        # Needed for display - Should not be used anywhere else. The cloud expects these so they must be filled
        payload[KEY_COMMAND] = computation.command.name
        assert KEY_MAX_SHOTS in computation.parameters, f"Missing '{KEY_MAX_SHOTS}' parameter"
        payload[KEY_MAX_SHOTS] = computation.parameters[KEY_MAX_SHOTS]

        return super().send(payload)

    def get_availability(self) -> int:
        try:
            availability = self._rpc_handler.get_job_availability()
            return availability["max_jobs_in_queue"] - availability["num_jobs_in_queue"]
        except HTTPError:
            get_logger().warn("Impossible to determine whether there is room for a new job")
            return 0
