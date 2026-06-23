# MIT License
#
# Copyright (c) 2026 Kipu Quantum GmbH
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

from perceval.runtime import ISession
from perceval.runtime.remote_processor import RemoteProcessor
from perceval.utils.logging import get_logger, channel

from .kipu_rpc_handler import KipuRPCHandler


class Session(ISession):
    """
    Kipu Quantum Hub session.

    :param platform_name: Hub backend id or alias (e.g. "quandela.sim.belenos")
    :param token: optional Kipu Personal Access Token (PAT); when omitted it is
        resolved from the environment or the `qhubctl login` config file
    :param organization_id: optional Kipu organization id; when omitted your
        personal account is used
    :param url: optional Hub base URL; when omitted the qhub-api client uses its
        own default Hub endpoint
    :param proxies: optional protocol->URL proxy mapping
    """

    def __init__(self, platform_name: str, token: str = None,
                 organization_id: str = None, url: str = None,
                 proxies: dict = None):
        if not platform_name:
            raise ValueError("platform_name cannot be None")
        self._platform_name = platform_name
        self._token = token
        self._organization_id = organization_id
        self._url = url
        self._proxies = proxies or {}
        get_logger().info(
            f"Creating Kipu Session to {self._url or 'default Hub endpoint'}",
            channel.general)

    def build_remote_processor(self) -> RemoteProcessor:
        """Build a RemoteProcessor wired to the Kipu Hub."""
        handler = KipuRPCHandler(
            platform_name=self._platform_name,
            url=self._url,
            token=self._token,
            organization_id=self._organization_id,
            proxies=self._proxies,
        )
        return RemoteProcessor(rpc_handler=handler)
