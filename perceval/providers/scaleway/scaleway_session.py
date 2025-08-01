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
from typing import Optional

from perceval.runtime import ISession
from perceval.runtime.remote_processor import RemoteProcessor
from perceval.utils.logging import get_logger, channel

from .scaleway_rpc_handler import RPCHandler


class Session(ISession):
    """
    :param platform_name: platform on which circuits will be executed

    :param project_id: UUID of the Scaleway Project the session is attached to

    :param token: authentication token required to access the Scaleway API

    :param deduplication_id: optional value, name mapping to a unique running session, allowing to share an alive session among multiple users

    :param max_idle_duration_s: optional value, duration in seconds that can elapsed without activity before the session terminates

    :param max_duration_s: optional value, duration in seconds for a session before it automatically terminates

    :param url: optional value, endpoint URL of the API

    :param proxies: optional value, dictionary mapping protocol to the URL of the proxy
    """

    def __init__(
        self,
        platform: str,
        project_id: str,
        token: str,
        max_idle_duration_s: int = 1200,
        max_duration_s: int = 3600,
        deduplication_id: Optional[str] = None,
        url: Optional[str] = None,
        proxies: Optional[dict[str, str]] = None,
        provider_name: Optional[str] = None,
    ) -> None:

        if not platform:
            raise Exception("platform_name cannot be None")

        if not project_id:
            raise Exception("project_id cannot be None")

        if not token:
            raise Exception("token cannot be None")

        if not isinstance(max_duration_s, int):
            raise TypeError("max_duration_s cannot be an int (ie: seconds)")

        if not isinstance(max_idle_duration_s, int):
            raise TypeError("max_idle_duration_s cannot be an int (ie: seconds)")

        self._deduplication_id = deduplication_id
        self._max_idle_duration_s = max_idle_duration_s
        self._max_duration_s = max_duration_s

        self._rpc_handler = RPCHandler(
            project_id=project_id,
            secret_key=token,
            url=url,
            proxies=proxies,
            platform_name=platform,
            provider_name=provider_name,
        )

        get_logger().info(f"Create Scaleway Session", channel.general)

    def build_remote_processor(self) -> RemoteProcessor:
        return RemoteProcessor(rpc_handler=self._rpc_handler)

    def start(self) -> None:
        self._rpc_handler.create_session(
            max_duration_s=self._max_idle_duration_s,
            max_idle_duration_s=self._max_idle_duration_s,
            deduplication_id=self._deduplication_id,
        )

    def stop(self) -> None:
        self._rpc_handler.terminate_session()
        get_logger().info("Stop Scaleway Session", channel.general)

    def delete(self) -> None:
        self._rpc_handler.delete_session()
        get_logger().info(
            "Stop (if not already) and revoke Scaleway Session", channel.general
        )
