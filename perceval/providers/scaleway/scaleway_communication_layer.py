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

from perceval.serialization import InputArchive, Serialization
from perceval.utils.logging import get_logger, channel

from .scaleway_rpc_handler import RPCHandler
from ..rpc_based_communication_layer import RPCBasedCommunicationLayer


class ScalewayCommunicationLayer(RPCBasedCommunicationLayer):

    def __init__(self,
                 platform_name: str,
                 project_id: str,
                 token: str,
                 max_idle_duration_s: int = 1200,
                 max_duration_s: int = 3600,
                 deduplication_id: str = None,
                 url: str = None,
                 proxies: dict[str, str] = None,
                 provider_name: str = None):

        self._deduplication_id = deduplication_id
        self._max_idle_duration_s = max_idle_duration_s
        self._max_duration_s = max_duration_s

        super().__init__(RPCHandler(
            project_id=project_id,
            secret_key=token,
            url=url,
            proxies=proxies,
            platform_name=platform_name,
            provider_name=provider_name,
        ))
        get_logger().info(f"Connected to Scaleway Cloud platform {platform_name}", channel.general)

    def start_session(self) -> None:
        self._rpc_handler.create_session(
            max_duration_s=self._max_idle_duration_s,
            max_idle_duration_s=self._max_idle_duration_s,
            deduplication_id=self._deduplication_id,
        )

    def stop_session(self) -> None:
        self._rpc_handler.terminate_session()
        get_logger().info("Stop Scaleway Session", channel.general)

    def delete_session(self) -> None:
        self._rpc_handler.delete_session()
        get_logger().info(
            "Stop (if not already) and revoke Scaleway Session", channel.general
        )

    @staticmethod
    def from_rpc(rpc_handler: RPCHandler):
        # We can't choose the session parameters here, so we use the default ones
        return ScalewayCommunicationLayer(platform_name=rpc_handler.name,
                                          project_id = rpc_handler._project_id,
                                          token = rpc_handler.headers["X-Auth-Token"],
                                          url = rpc_handler.url,
                                          proxies = rpc_handler.proxies,
                                          provider_name = rpc_handler._provider_name)


def _load_scaleway_communication_layer(
    communication_layer: ScalewayCommunicationLayer,
    archive: InputArchive,
    members,
    version: int,
):
    values = {name: index for name, index in members}
    RPCBasedCommunicationLayer.__init__(communication_layer, archive.create(values.pop("_rpc_handler")))
    archive.load_attr(communication_layer, list(values.items()))


Serialization.register_class(
    ScalewayCommunicationLayer,
    class_serial_members_write=lambda communication_layer, archive: archive.save_attr(
        communication_layer, ["_rpc_handler", "_deduplication_id", "_max_idle_duration_s", "_max_duration_s"]
    ),
    class_serial_members_read=_load_scaleway_communication_layer,
)
