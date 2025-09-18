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

from perceval.runtime import ISession
from perceval.runtime.remote_processor import RemoteProcessor
from perceval.runtime.remote_config import QUANDELA_CLOUD_URL
from perceval.runtime.rpc_handler import RPCHandler
from perceval.utils.logging import get_logger, channel


class Session(ISession):
    """
    Quandela Cloud session

    :param platform_name: Name of an available platform on Quandela Cloud (e.g. "sim:slos")
    :param token: A valid authentication token to use within the session
    :param url: URL prefix for the endpoint to connect to. If omitted the Quandela Cloud endpoints will be used.
    """
    def __init__(self, platform_name: str, token: str, url: str = None):
        self._platform_name = platform_name
        self._token = token
        self._url = url or QUANDELA_CLOUD_URL
        get_logger().info(f"Creating Quandela Session to {self._url}", channel.general)

    def build_remote_processor(self) -> RemoteProcessor:
        """Build a remote processor from the session information"""
        return RemoteProcessor(rpc_handler=RPCHandler(self._platform_name, self._url, self._token))
