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
import urllib
import time
from datetime import datetime, timedelta
import json

from enum import Enum
import requests
from requests import HTTPError

_ENDPOINT_URL = "https://api.scaleway.com"
_PROVIDER_NAME = "quandela"
_ENDPOINT_PLATFORM = "/platforms"
_ENDPOINT_JOB = "/jobs"


class JobStatus(Enum):
    """JobStatus Enum"""

    COMPLETED = "completed"
    ERROR = "error"


class RPCHandler:
    """RPCHandler Scaleway """

    def __init__(self, project_id, token, url: str = _ENDPOINT_URL) -> None:
        self._token = token
        self.project_id = project_id
        self.url = url
        self.name = None
        self.session_id = None

        self.headers = {
            "X-Auth-Token": token,
        }

    def fetch_platform_details(self) -> dict:
        endpoint = f"{self._build_endpoint(_ENDPOINT_PLATFORM)}?providerName={_PROVIDER_NAME}&name={urllib.parse.quote_plus(self.name)}"
        resp = requests.get(endpoint, headers=self.headers)

        resp.raise_for_status()
        resp_dict = resp.json()

        if "platforms" not in resp_dict:
            raise HTTPError(f"platforms '{self.name}' not found")

        platform_dict = resp_dict["platforms"][0]
        platform_dict["specs"] = json.loads(platform_dict.get("metadata", {}))

        return platform_dict

    def create_job(self, payload) -> str:
        scw_payload = {
            "name": payload.get("job_name"),
            "circuit": {"percevalCircuit": json.dumps(payload.get("payload", {}))},
            "project_id": self.project_id,
            "session_id": self.session_id,
        }

        endpoint = f"{self.url}{_ENDPOINT_JOB}"
        request = requests.post(endpoint, headers=self.headers, json=scw_payload)

        try:
            request.raise_for_status()
            request_dict = request.json()

            self.instance_id = request_dict["id"]
        except Exception:
            raise HTTPError(request.json())

        return request_dict["id"]

    def cancel_job(self, job_id: str) -> None:
        endpoint = f"{self._build_endpoint(_ENDPOINT_JOB)}/{job_id}/cancel"
        request = requests.post(endpoint, headers=self.headers)
        request.raise_for_status()

    def get_job_status(self, job_id: str) -> dict:
        endpoint = f"{self._build_endpoint(_ENDPOINT_JOB)}/{job_id}"

        # requests may throw an IO Exception, let the user deal with it
        resp = requests.get(endpoint, headers=self.headers)
        resp.raise_for_status()

        resp_dict = resp.json()

        creation_datetime = datetime.fromisoformat(
            resp_dict.get("created_at")
        ).timestamp()
        start_time = self.__get_start_time(resp_dict.get("started_at"))
        duration = self.__get_duration(start_time)
        status = resp_dict.get("status")

        return {
            "creation_datetime": creation_datetime,
            "duration": duration,
            "failure_code": None,
            "last_intermediate_results": None,
            "msg": "ok",
            "progress": self.__get_progress_by_status(status),
            "progress_message": resp_dict.get("progress_message"),
            "start_time": start_time,
            "status": status,
            "status_message": self.__get_status_message(
                status, resp_dict.get("result_distribution")
            ),
        }

    def get_job_results(self, job_id: str) -> dict:
        endpoint = f"{self._build_endpoint(_ENDPOINT_JOB)}/{job_id}"

        # requests may throw an IO Exception, let the user deal with it
        resp = requests.get(endpoint, headers=self.headers)
        resp.raise_for_status()

        resp_dict = resp.json()

        duration = self.__get_duration(
            self.__get_start_time(resp_dict.get("started_at"))
        )

        return {
            "duration": duration,
            "intermediate_results": [],
            "job_id": resp_dict.get("id"),
            "results": json.dumps(resp_dict.get("result_distribution", {})),
            "results_type": None,
        }

    def _build_endpoint(self, endpoint) -> str:
        return f"{self.url}{endpoint}"

    def __get_start_time(self, started_at: str | None) -> float | None:
        return datetime.fromisoformat(started_at).timestamp() if started_at else None

    def __get_duration(self, start_time: float | None) -> int | None:
        return (
            timedelta(seconds=time.time() - start_time).seconds if start_time else None
        )

    def __get_progress_by_status(self, status: str) -> float:
        if status == JobStatus.COMPLETED.value:
            return 1.0
        return 0.0

    def __get_status_message(self, status, result_distribution) -> str | None:
        if status == JobStatus.ERROR.value:
            return result_distribution
        return None
