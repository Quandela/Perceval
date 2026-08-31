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

from perceval.serialization import InputArchive, Serialization
from perceval.utils.logging import get_logger, channel

from .rpc_handler import RPCHandler
from ..rpc_based_communication_layer import RPCBasedCommunicationLayer


class QuandelaCommunicationLayer(RPCBasedCommunicationLayer):

    def __init__(self, name: str, token: str, url: str, proxies: dict[str, str] = None):
        super().__init__(RPCHandler(name, url, token, proxies))
        get_logger().info(f"Connected to Cloud platform {name}", channel.general)

    @staticmethod
    def from_rpc(rpc_handler: RPCHandler):
        return QuandelaCommunicationLayer(rpc_handler.name, rpc_handler.token, rpc_handler.url, rpc_handler.proxies)

    def get_availability(self) -> int:
        try:
            availability = self._rpc_handler.get_job_availability()
            return availability["max_jobs_in_queue"] - availability["num_jobs_in_queue"]
        except HTTPError:
            get_logger().warn("Impossible to determine whether there is room for a new job")
            return 0


def _load_quandela_communication_layer(
    communication_layer: QuandelaCommunicationLayer,
    archive: InputArchive,
    members,
    version: int,
):
    RPCBasedCommunicationLayer.__init__(communication_layer, archive.create(members[0][1]))


Serialization.register_class(
    QuandelaCommunicationLayer,
    class_serial_members_write=lambda communication_layer, archive: archive.save_attr(
        communication_layer, ["_rpc_handler"]
    ),
    class_serial_members_read=_load_quandela_communication_layer,
)
