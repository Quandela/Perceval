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

from time import time
from typing import TypeAlias

from requests import HTTPError

from .platform_specs import PlatformSpecs
from .remote_computer import RemoteComputer, CommunicationLayer
from .remote_config import RemoteConfig
from .job_status import RunningStatus
from .remote_processor import PERFS_KEY
from .rpc_handler import RPCHandler

from perceval.utils.logging import get_logger, channel
from perceval.serialization import deserialize, serialize

AsyncGetter: TypeAlias = str


# TODO: add robustness to communication failures
class QuandelaCommunicationLayer(CommunicationLayer):

    MINIMUM_FETCH_INTERVAL_SECONDS = 5

    def __init__(self, name: str, token: str, url: str, proxies: dict[str, str]):
        self.name = name
        self.token = token
        self.url = url
        self.proxies = proxies
        self._specs = PlatformSpecs()
        self._status: str = None
        self._perfs: dict[str, str] = {}
        self._last_fetch_time = None
        self._rpc_handler = RPCHandler(name, url, token, proxies)

        self.fetch_data()
        get_logger().info(f"Connected to Cloud platform {self.name}", channel.general)

    def fetch_data(self):
        # Quandela specific: the same endpoint gives the specs, perfs and platform status
        if self._last_fetch_time is None or time() - self._last_fetch_time > self.MINIMUM_FETCH_INTERVAL_SECONDS:
            try:
                platform_details = self._rpc_handler.fetch_platform_details()
            except HTTPError as e:
                if not len(self._specs):  # throw only the first time
                    raise HTTPError(f"Error while fetching platform details: {e}") from None
                else:
                    get_logger().warn(f"Error while fetching platform details: {e}")
                    return

            self._status = platform_details.get("status")
            platform_specs = deserialize(platform_details['specs'], strict=False)
            self._specs.update(platform_specs)  # No verification here, we suppose every check was made by the platform
            self._specs["type"] = platform_details.get('type', "simulator")
            if PERFS_KEY in platform_details:
                self._perfs.update(platform_details[PERFS_KEY])

            self._last_fetch_time = time()

    def get_specs(self) -> PlatformSpecs:
        return self._specs

    @property
    def is_available(self) -> bool:
        try:
            availability = self._rpc_handler.get_job_availability()
            return availability["max_jobs_in_queue"] > availability["num_jobs_in_queue"]
        except HTTPError:
            get_logger().warn("Impossible to determine whether there is room for a new job")
            return False

    def send(self, payload: dict) -> AsyncGetter:
        return self._rpc_handler.create_job(serialize(payload))

    def get_results(self, async_getter: AsyncGetter) -> dict:
        return self._rpc_handler.get_job_results(async_getter)

    def get_status(self, async_getter: AsyncGetter) -> RunningStatus:
        return RunningStatus.from_server_response(self._rpc_handler.get_job_status(async_getter)["TODO"])

    def get_performances(self) -> dict:
        self.fetch_data()
        return self._perfs

    def get_commands(self) -> list[str]:
        return self._specs.available_commands

    def cancel(self, async_getter: AsyncGetter) -> None:
        self._rpc_handler.cancel_job(async_getter)


# Make this a simple method ? Answer will depend on if we need to add/modify methods for quandela
class QuandelaComputer(RemoteComputer):

    def __init__(self,
                 name: str = None,
                 token: str = None,
                 url: str = None,
                 proxies: dict[str,str] = None,
                 communication_layer: QuandelaCommunicationLayer = None):

        if communication_layer is not None:  # When a com_layer object is passed, name, token and url are expected to be None
            self.name = communication_layer.name  # Here, we are mixing the experiment name and the Processor name
            if name is not None and name != communication_layer.name:
                get_logger().warn(
                    f"Initialised a RemoteProcessor with two different platform names ({self.name} vs {name})", channel.user)
        else:
            remote = RemoteConfig()
            if name is None:
                raise ValueError("Parameter 'name' must have a value")
            if token is None:
                token = remote.get_token()
            if not token:
                raise ConnectionError("No token found")
            if url is None:
                url = remote.get_url()
            if proxies is None:
                proxies = remote.get_proxies()
            self.name = name
            communication_layer = QuandelaCommunicationLayer(name, url, token, proxies)

        super().__init__(communication_layer)
