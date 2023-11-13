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
from .scaleway_rpc_handler import RPCHandler

import requests
from requests import HTTPError

_ENDPOINT_SESSION = "/sessions"


class Session(ISession):
    """Session Scaleway"""

    def __init__(
        self,
        platform: str,
        rpc_handler: RPCHandler,
        deduplication_id: str = "",
        max_idle_duration: str = "120s",
        max_duration: str = "360s",
    ) -> None:
        self._platform = platform
        self._deduplication_id = deduplication_id
        self._max_idle_duration = max_idle_duration
        self._max_duration = max_duration
        self._session_id = None

        rpc_handler.name = platform
        self._rpc_handler = rpc_handler

        self._url = rpc_handler.url
        self._headers = rpc_handler.headers

    def build_remote_processor(self) -> RemoteProcessor:
        return RemoteProcessor(rpc_handler=self._rpc_handler)

    def start(self) -> None:
        platform = self.__fetch_platform_details()

        payload = {
            "project_id": self._rpc_handler.project_id,
            "platform_id": platform.get("id"),
            "deduplication_id": self._deduplication_id,
            "max_duration": self._max_duration,
            "max_idle_duration": self._max_idle_duration,
        }

        endpoint = f"{self._url}{_ENDPOINT_SESSION}"
        request = requests.post(endpoint, headers=self._headers, json=payload)

        try:
            request.raise_for_status()
            request_dict = request.json()

            self._session_id = request_dict["id"]
            self._rpc_handler.session_id = self._session_id
        except Exception:
            raise HTTPError(request.json())

    def stop(self) -> None:
        endpoint = f"{self._url}{_ENDPOINT_SESSION}/{self._session_id}"
        request = requests.delete(endpoint, headers=self._headers)

        request.raise_for_status()

    def __fetch_platform_details(self) -> dict:
        return self._rpc_handler.fetch_platform_details()
