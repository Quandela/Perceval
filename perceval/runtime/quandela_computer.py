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

from .command import Command
from .platform_specs import PlatformSpecs
from .remote_computer import RemoteComputer, CommunicationLayer, RemoteId, _RemoteGetter
from .remote_config import RemoteConfig
from .job_status import RunningStatus, JobStatus
from .remote_job import _retrieve_from_response
from .remote_processor import PERFS_KEY
from .rpc_handler import RPCHandler
from .computation import Computation
from .payload_generator import PayloadGenerator
from .payload_updater import PayloadUpdater

from perceval.serialization import deserialize, serialize
from perceval.utils import ContextManager
from perceval.utils.logging import get_logger, channel
from perceval.utils.constants import KEY_JOB_CONTEXT, KEY_RESULT_MAPPING, KEY_RESULTS_LIST, KEY_MAPPING_PARAMETERS, \
    KEY_ITERATION, KEY_RESULTS, KEY_PLATFORM_NAME, KEY_JOB_NAME, KEY_JOB_GROUP_NAME, KEY_COMMAND, KEY_MAX_SHOTS


class QuandelaCommunicationLayer(CommunicationLayer):

    MINIMUM_FETCH_INTERVAL = 5
    _MAX_ERROR = 6

    def __init__(self, name: str, token: str, url: str, proxies: dict[str, str] = None):
        self.name = name
        self.token = token
        self.url = url
        self.proxies: dict[str, str] = proxies if proxies is not None else {}
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
            self._specs = PlatformSpecs(platform_specs)
            self._specs["type"] = platform_details.get('type', "simulator")
            if PERFS_KEY in platform_details:
                self._perfs.update(platform_details[PERFS_KEY])

            self._last_fetch_time = time.time()

    def get_specs(self) -> PlatformSpecs:
        return self._specs

    def send(self, payload: dict) -> RemoteId:
        computation = PayloadGenerator.get_computation(payload)

        # Needed for display - Should not be used anywhere else. The cloud expects these so they must be filled
        payload[KEY_COMMAND] = computation.command.name
        payload[KEY_MAX_SHOTS] = computation.parameters[KEY_MAX_SHOTS]

        if "commands" not in self._specs:  # We have a worker that knows only payloads up to version 1
            # Using self._specs is a bit of a trick, since internally,
            # we only needs the argument to have "available_commands" when downgrading to version 1
            # This might not be true anymore if we introduce a version 3 someday
            payload = PayloadUpdater.update_payload(payload, self._specs, target_payload_version = 1)

        global_data = PayloadGenerator.generate_global_data(payload,
                                                            {KEY_PLATFORM_NAME: self._rpc_handler.name,
                                                             KEY_JOB_NAME: computation.job_name,
                                                             KEY_JOB_GROUP_NAME: computation.job_group_name})

        return self._rpc_handler.create_job(serialize(global_data))

    def get_results(self, remote_id: RemoteId) -> dict:
        try:
            response = self._rpc_handler.get_job_results(remote_id)
        except HTTPError as e:
            raise HTTPError(f"Error while retrieving job results: {e}") from None
        # Note: this is not KEY_RESULTS since this is the cloud response format, not perceval response format
        results = deserialize(json.loads(response["results"]), strict=False)
        if not isinstance(results, dict):
            return {}

        # TODO: remove (deprecated since 1.3, old return format)
        if KEY_JOB_CONTEXT in results and KEY_RESULT_MAPPING in results[KEY_JOB_CONTEXT]:
            path_parts = results[KEY_JOB_CONTEXT][KEY_RESULT_MAPPING]
            get_logger().info(f"Converting job {remote_id} results with {path_parts[1]}", channel.general)
            module = __import__(path_parts[0], fromlist=path_parts[1])
            result_mapping_function = getattr(module, path_parts[1])
            # retrieve delta parameters from the response
            delta_parameters = results[KEY_JOB_CONTEXT].get(KEY_MAPPING_PARAMETERS, {})
            if KEY_RESULTS_LIST in results:
                for res in results[KEY_RESULTS_LIST]:
                    mapping_args = {key: res[KEY_ITERATION].get(key, val) for key, val in delta_parameters.items()}
                    res[KEY_RESULTS] = result_mapping_function(res[KEY_RESULTS], **mapping_args)
            else:
                results[KEY_RESULTS] = result_mapping_function(results[KEY_RESULTS], **delta_parameters)
        return results

    def _handle_status_error(self, error: Exception, remote_id: RemoteId, refresh_errors: int):
        """
        Handle a potentially non-blocking error
        After _MAX_ERROR errors in a row, the exception is raised
        """
        if refresh_errors == self._MAX_ERROR:
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

    def get_commands(self) -> list[Command]:
        return self._specs.commands

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


class QuandelaComputer(RemoteComputer):
    _communication_layer: QuandelaCommunicationLayer  # Used for type hinting only

    WARN_INTERVAL = 1800
    INFO_INTERVAL = 10

    def __init__(self,
                 name: str,
                 token: str = None,
                 url: str = None,
                 proxies: dict[str,str] = None):
        """
        A Computer meant to access Quandela remote services.

        All parameters but the name of the target platform have to be explicitly named to be set.

        :param name: Platform name.
        :param token: Token value to authenticate the user. If not provided, it is taken from the stored RemoteConfig
        :param url: Base URL for the Cloud API to connect to
        :param proxies: Dictionary mapping protocol to the URL of the proxy
        """

        remote = RemoteConfig()
        if token is None:
            token = remote.get_token()
        if not token:
            raise ConnectionError("No token found")
        if url is None:
            url = remote.get_url()
        if proxies is None:
            proxies = remote.get_proxies()
        self.name = name
        communication_layer = QuandelaCommunicationLayer(name, token, url, proxies)

        super().__init__(communication_layer)
        self._available_jobs = communication_layer.get_availability()

    def validate_single(self, computation: Computation) -> None:
        super().validate_single(computation)
        assert KEY_MAX_SHOTS in computation.parameters, f"Missing '{KEY_MAX_SHOTS}' parameter"

    def _take_resource(self):
        start = time.time()
        start_warn = time.time()
        start_info = start_warn
        while self._available_jobs <= 0:
            self._available_jobs = self._communication_layer.get_availability()
            time.sleep(1)
            if time.time() - start_warn > self.WARN_INTERVAL:
                start_warn = time.time()
                get_logger().warn(f"Couldn't find a way to send any job for {int(start_warn - start)} seconds - queue is full")
            elif time.time() - start_info > self.INFO_INTERVAL:
                start_info = time.time()
                get_logger().info(f"Couldn't find a way to send any job for {int(start_info - start)} seconds - queue is full")

        self._available_jobs -= 1

    def _release_resource(self):
        self._available_jobs += 1

    def _reserve_resource(self) -> ContextManager:
        return ContextManager(self._take_resource, self._release_resource)

    def _execute_command_async(self, computation: Computation) -> _RemoteGetter:
        self._take_resource()
        return super()._execute_command_async(computation)
