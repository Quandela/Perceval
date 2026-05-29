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
import json
import time

from requests import HTTPError

from .platform_specs import PlatformSpecs
from .remote_computer import RemoteComputer, CommunicationLayer, RemoteId
from .remote_config import RemoteConfig
from .job_status import RunningStatus, JobStatus
from .remote_job import _retrieve_from_response
from .remote_processor import PERFS_KEY
from .rpc_handler import RPCHandler
from .computation import Computation
from .payload_generator import PayloadGenerator

from perceval.utils.logging import get_logger, channel
from perceval.serialization import deserialize, serialize
from perceval.utils import ContextManager


class QuandelaCommunicationLayer(CommunicationLayer):

    MINIMUM_FETCH_INTERVAL = 5
    _MAX_ERROR = 5

    def __init__(self, name: str, token: str, url: str, proxies: dict[str, str]):
        self.name = name
        self.token = token
        self.url = url
        self.proxies = proxies
        self._specs = PlatformSpecs()
        self._status: str = ""
        self._perfs: dict[str, str] = {}
        self._last_fetch_time = None
        self._rpc_handler = RPCHandler(name, url, token, proxies)

        self.fetch_data()
        get_logger().info(f"Connected to Cloud platform {self.name}", channel.general)

    def fetch_data(self):
        # Quandela specific: the same endpoint gives the specs, perfs and platform status
        if self._last_fetch_time is None or time.time() - self._last_fetch_time > self.MINIMUM_FETCH_INTERVAL:
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

            self._last_fetch_time = time.time()

    def get_specs(self) -> PlatformSpecs:
        return self._specs

    def send(self, payload: dict) -> RemoteId:
        # TODO: how to be compatible with old format to receive names?
        computation = payload["computation"]

        global_data = PayloadGenerator.generate_global_data(payload,
                                                            platform_name=self._rpc_handler.name,
                                                            job_name=computation.job_name,
                                                            job_group_name=computation.job_group_name)

        return self._rpc_handler.create_job(serialize(global_data))

    def get_results(self, remote_id: RemoteId) -> dict:
        try:
            response = self._rpc_handler.get_job_results(remote_id)
        except HTTPError as e:
            raise HTTPError(f"Error while retrieving job results: {e}") from None
        results = deserialize(json.loads(response['results']), strict=False)
        if not isinstance(results, dict):
            return {}

        # TODO: remove (deprecated since 1.3, old return format)
        if "job_context" in results and 'result_mapping' in results["job_context"]:
            path_parts = results["job_context"]["result_mapping"]
            get_logger().info(f"Converting job {remote_id} results with {path_parts[1]}", channel.general)
            module = __import__(path_parts[0], fromlist=path_parts[1])
            result_mapping_function = getattr(module, path_parts[1])
            # retrieve delta parameters from the response
            delta_parameters = results["job_context"].get("mapping_delta_parameters", {})
            if "results_list" in results:
                for res in results["results_list"]:
                    mapping_args = {key: res["iteration"].get(key, val) for key, val in delta_parameters.items()}
                    res["results"] = result_mapping_function(res['results'], **mapping_args)
            else:
                results["results"] = result_mapping_function(results["results"], **delta_parameters)
        return results

    def _handle_status_error(self, error: Exception, remote_id: RemoteId, refresh_errors: int):
        """
        Handle a potentially non-blocking error
        After _MAX_ERROR errors in a row, the exception is raised
        """
        if refresh_errors + 1 == self._MAX_ERROR:
            get_logger().error(f"Reached max number of HTTP errors in a row when updating job {remote_id} status.",
                               channel.general)
            raise error
        if isinstance(error, HTTPError):
            error_code = error.response.status_code
            if error_code in [
                408,  # Time-out
                409,  # Conflict in the current state of the resource
                421,  # Misdirected request
                423,  # Resource locked
                429   # Too many requests
            ]:
                get_logger().error(f"Got HTTP error {error_code} when updating job {remote_id} status. Ignoring...",
                                   channel.general)
            else:  # If the status code is any other error, it is considered unrecoverable
                raise error

    def get_job_status(self, remote_id: RemoteId, refresh_errors: int = 0) -> JobStatus | None:
        try:
            response = self._rpc_handler.get_job_status(remote_id)
        except (HTTPError, ConnectionError) as error:
            self._handle_status_error(error, remote_id, refresh_errors)
            return None

        job_status = JobStatus()
        job_status.status = RunningStatus.from_server_response(_retrieve_from_response(response, 'status'))
        if job_status.running:
            job_status.update_progress(_retrieve_from_response(response, 'progress', 0., float),
                                       _retrieve_from_response(response, 'progress_message'))
        elif job_status.failed:
            job_status._stop_message = _retrieve_from_response(response, 'status_message')

        self._extract_job_times(job_status, response)
        return job_status

    @staticmethod
    def _extract_job_times(status: JobStatus, response: dict) -> None:
        creation_datetime = _retrieve_from_response(response, 'creation_datetime', 0., float)

        start_datetime = 0.
        if not status.waiting:
            start_datetime = _retrieve_from_response(response, 'start_time', start_datetime, float)

        duration = 0
        if status.completed:
            duration = _retrieve_from_response(response, 'duration', duration, int)
        status.update_times(creation_datetime, start_datetime, duration)

    def get_performances(self) -> dict:
        self.fetch_data()
        return self._perfs

    def get_commands(self) -> list[str]:
        return self._specs.available_commands

    def get_remote_status(self) -> str:
        self.fetch_data()
        return self._status

    def cancel(self, remote_id: RemoteId) -> None:
        try:
            self._rpc_handler.cancel_job(remote_id)
        except HTTPError as e:
            raise HTTPError(f"Error while trying to cancel job: {e}") from None

    def get_availability(self) -> int:
        """
        :return: The number of jobs available in the queue
        """
        try:
            availability = self._rpc_handler.get_job_availability()
            return availability["max_jobs_in_queue"] - availability["num_jobs_in_queue"]
        except HTTPError:
            get_logger().warn("Impossible to determine whether there is room for a new job")
            return 0

    def _check_max_shots_samples_validity(self):
        # TODO: this should be moved to the Computer
        p = self._request_data['payload']
        if "max_samples" in p and "max_shots" in p:
            if p["max_samples"] > p["max_shots"]:
                get_logger().warn(f"Lowered 'max_samples' from user defined value ({p['max_samples']}) to 'max_shots' value ({p['max_shots']}) for consistency.",
                                  channel.user)
                p["max_samples"] = p["max_shots"]


class QuandelaComputer(RemoteComputer):
    _communication_layer: QuandelaCommunicationLayer  # Used for type hinting only

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
        self._available_jobs = communication_layer.get_availability()

    def _take_resource(self):
        while self._available_jobs == 0:
            self._available_jobs = self._communication_layer.get_availability()
            time.sleep(1)
        self._available_jobs -= 1

    def _release_resource(self):
        self._available_jobs += 1

    def _reserve_resource(self) -> ContextManager:
        return ContextManager(self._take_resource, self._release_resource)

    def _execute_command_async(self, computation: Computation) -> int:
        payload = self.prepare_payload(computation)
        self._take_resource()
        return self._communication_layer.send(payload)
