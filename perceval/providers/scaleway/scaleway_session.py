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

_ENDPOINT_URL = "https://api.scaleway.com/qaas/v1alpha1"
_ENDPOINT_SESSION = "/sessions"


class Session(ISession):
    """Session Scaleway

        Attributes:
            platform (str): The `platform` parameter is a string that represents the platform on which the
            circuit will be execute

            project_id (str): The `project_id` parameter is a string that represents the UUID of the Scaleway project
            you want your sessions & jobs to be attached with

            token (str): The `token` parameter is a string that represents the authentication token
            required to access the Scaleway API

            deduplication_id (str, optional):  Value that points on a unique running session.
            If both Perceval programs (with same credentials) are targetting the same deduplication_id, it will returns the same session.
            It is usefull to share session amongs same users. The deduplication_id works only on alive session.

            max_idle_duration (int, optional): The `max_idle_duration` parameter is an int that represents the
            maximum duration of idle time allowed for a session. It specifies the amount of time that can
            elapse without any activity before the session is considered idle. The default value is 1200,
            which means 1200 seconds (or 20minutes)

            max_duration (int, optional): The `max_duration` parameter is a int that represents the maximum
            duration for a session. It specifies the maximum amount of time that a session can remain active
            before it is automatically terminated. The default value is 3600, which means 3600 seconds (or 1hour)

            url (str, optional): The `url` parameter is a string that represents the endpoint URL for the API. It is
            optional and has a default value of `_ENDPOINT_URL`
    """

    def __init__(
        self,
        platform: str,
        project_id: str,
        token: str,
        deduplication_id: str = "",
        max_idle_duration: int = 1200,
        max_duration: int = 3600,
        url: str = _ENDPOINT_URL,
    ) -> None:

        self._token = token
        self._project_id = project_id
        self._url = url
        self._platform = platform
        self._deduplication_id = deduplication_id
        self._max_idle_duration = self._int_duration(max_idle_duration, 'max_idle_duration')
        self._max_duration = self._int_duration(max_duration, 'max_duration')

        self._session_id = None

        self._headers = {
            "X-Auth-Token": token,
        }

        self._rpc_handler = self._build_rpc_handler()

    def build_remote_processor(self) -> RemoteProcessor:
        return RemoteProcessor(rpc_handler=self._rpc_handler)

    def start(self) -> None:
        platform = self._fetch_platform_details()

        payload = {
            "project_id": self._project_id,
            "platform_id": platform.get("id"),
            "deduplication_id": self._deduplication_id,
            "max_duration": self._to_string_duration(self._max_duration),
            "max_idle_duration": self._to_string_duration(self._max_idle_duration),
        }

        endpoint = f"{self._url}{_ENDPOINT_SESSION}"
        request = requests.post(endpoint, headers=self._headers, json=payload)

        try:
            request.raise_for_status()
            request_dict = request.json()

            self._session_id = request_dict["id"]
            self._rpc_handler.set_session_id(self._session_id)
        except Exception:
            raise HTTPError(request.json())

    def stop(self) -> None:
        endpoint = f"{self._url}{_ENDPOINT_SESSION}/{self._session_id}"
        request = requests.delete(endpoint, headers=self._headers)

        request.raise_for_status()

    def _fetch_platform_details(self) -> dict:
        return self._rpc_handler.fetch_platform_details()

    def _to_string_duration(self, duration: int) -> str:
        return f"{duration}s"

    def _int_duration(self, duration, name: str) -> int:
        if isinstance(duration, int):
            return duration
        raise TypeError(f"{name} must be an int")

    def _build_rpc_handler(self) -> RPCHandler:
        return RPCHandler(
            project_id=self._project_id,
            headers=self._headers,
            name=self._platform,
            url=self._url
        )
