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
from typing import TypeVar, Type

from requests import HTTPError

from perceval.serialization import InputArchive, Serialization, deserialize, serialize, OutputArchive
from perceval.utils.constants import KEY_JOB_NAME, KEY_JOB_CONTEXT, KEY_RESULT_MAPPING, \
    KEY_MAPPING_PARAMETERS, KEY_RESULTS_LIST, KEY_ITERATION, KEY_RESULTS, KEY_PLATFORM_NAME, KEY_JOB_GROUP_NAME, \
    KEY_COMMAND, KEY_MAX_SHOTS, KEY_MAX_SAMPLES
from perceval.utils.logging import channel, get_logger

from perceval.runtime import ExecutionStatus, RunningStatus, Command, PlatformSpecs, PayloadUpdater, PayloadGenerator, \
    CommunicationLayer


PERFS_KEY = "perfs"
T = TypeVar('T')

RemoteId = TypeVar("RemoteId")

def _retrieve_from_response(response: dict, field: str, default_value: T = '', value_type: Type[T] = str) -> T:
    if field not in response:
        get_logger().error(f"Missing field '{field}' from server response. Using default value {default_value}.", channel.general)
        return default_value
    try:
        result = value_type(response[field])
    except (ValueError, TypeError):
        get_logger().error(f"The field '{field}' from server response contains the wrong value '{response[field]}'. Using default value {default_value}.", channel.general)
        result = default_value
    return result


class RPCBasedCommunicationLayer(CommunicationLayer):
    MINIMUM_FETCH_INTERVAL = 5
    _MAX_ERROR = 6

    # Use duck-typing for RPCHandler. See the quandela RPCHandler for an example
    def __init__(self, rpc_handler):
        self._rpc_handler = rpc_handler

        self._specs = PlatformSpecs()
        self._status: str = ""
        self._perfs: dict[str, str] = {}
        self._last_fetch_time = None

        self.fetch_data()

    def fetch_data(self):
        # RPCHandler specific: the same method gives the specs, perfs and platform status
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

    @staticmethod
    def _serialize(obj):
        archive = OutputArchive()
        Serialization.serialize(obj, archive)
        return archive.to_text()  # Use other format ? Compress ?

    def send(self, payload: dict) -> RemoteId:
        computation = PayloadGenerator.get_computation(payload)

        # Needed for display - Should not be used anywhere else. The cloud expects these so they must be filled
        payload[KEY_COMMAND] = computation.command.name
        assert KEY_MAX_SHOTS in computation.parameters, f"Missing '{KEY_MAX_SHOTS}' parameter"
        payload[KEY_MAX_SHOTS] = computation.parameters[KEY_MAX_SHOTS]
        payload[KEY_MAX_SAMPLES] = computation.parameters.get(KEY_MAX_SAMPLES, 0)

        if "commands" not in self._specs:  # We have a worker that knows only payloads up to version 1
            # Using self._specs is a bit of a trick, since internally,
            # we only needs the argument to have "available_commands" when downgrading to version 1
            # This might not be true anymore if we introduce a version 3 someday
            payload = PayloadUpdater.update_payload(payload, self._specs, target_payload_version=1)

        else:
            # We serialize the payload here, using the new serialization system - Needed to serialize Computation
            cloud_needed_fields = [KEY_COMMAND, KEY_MAX_SHOTS, KEY_MAX_SAMPLES]
            for key, value in payload.items():
                if key not in cloud_needed_fields:
                    payload[key] = self._serialize(value)

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
                429  # Too many requests
            ]:
                get_logger().error(f"Got HTTP error {error_code} when updating job {remote_id} status. Ignoring...",
                                   channel.general)
            else:  # If the status code is any other error, it is considered unrecoverable
                raise error

    def get_job_status(self, remote_id: RemoteId, refresh_errors: int = 0) -> ExecutionStatus | None:
        try:
            response = self._rpc_handler.get_job_status(remote_id)
        except (HTTPError, ConnectionError) as error:
            self._handle_status_error(error, remote_id, refresh_errors)
            return None

        job_status = ExecutionStatus()
        job_status.status = RunningStatus.from_server_response(_retrieve_from_response(response, 'status'))
        if job_status.running or job_status.completed:
            job_status.update_progress(_retrieve_from_response(response, 'progress', 0., float),
                                       _retrieve_from_response(response, 'progress_message'))
        if job_status.failed:
            job_status._stop_message = _retrieve_from_response(response, 'status_message')

        self._extract_job_times(job_status, response)
        return job_status

    @staticmethod
    def _extract_job_times(status: ExecutionStatus, response: dict) -> None:
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
        return 1


def _load_rpc_communication_layer(
    communication_layer: RPCBasedCommunicationLayer,
    archive: InputArchive,
    members,
    version: int,
):
    archive.load_attr(communication_layer, members)
    communication_layer.fetch_data()  # Will not fetch if we are below MINIMUM_FETCH_INTERVAL


Serialization.register_class(
    RPCBasedCommunicationLayer,
    class_serial_members_write=lambda communication_layer, archive: archive.save_attr(
        communication_layer, ["_rpc_handler", "_specs", "_perfs", "_status", "_last_fetch_time"]
    ),
    class_serial_members_read=_load_rpc_communication_layer,
)
