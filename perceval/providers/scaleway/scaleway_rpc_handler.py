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
import requests
import json

from datetime import datetime, timedelta
from requests import HTTPError

from perceval.utils.logging import get_logger, channel

_ENDPOINT_PLATFORM = "/qaas/v1alpha1/platforms"
_ENDPOINT_JOB = "/qaas/v1alpha1/jobs"
_ENDPOINT_SESSION = "/qaas/v1alpha1/sessions"

_DEFAULT_URL = "https://api.scaleway.com"
_DEFAULT_PLATFORM_PROVIDER = "quandela"


class RPCHandler:
    """Stateful (1 session at a time) RPCHandler for Scaleway Cloud provider"""

    def __init__(
        self,
        project_id: str,
        secret_key: str,
        url: str,
        proxies: dict,
        platform_name: str,
        provider_name: str,
    ):
        self._project_id = project_id
        self._url = url or _DEFAULT_URL
        self._proxies = proxies or dict()
        self._session_id = None
        self._platform_name = platform_name
        self._headers = {
            "X-Auth-Token": secret_key,
        }
        self._provider_name = provider_name or _DEFAULT_PLATFORM_PROVIDER
        self._platform_id = self.get_platform(
            platform_name=self._platform_name, provider_name=self._provider_name
        )["id"]

    @property
    def name(self) -> str:
        return self._platform_name

    @property
    def url(self) -> str:
        return self._url

    @property
    def headers(self) -> dict:
        return self._headers

    @property
    def proxies(self) -> dict:
        return self._proxies

    def fetch_platform_details(self) -> dict:
        return self.get_platform(self._platform_name, self._provider_name)

    def get_platform(self, platform_name: str, provider_name: str) -> dict:
        endpoint = f"{self.__build_endpoint(_ENDPOINT_PLATFORM)}?providerName={provider_name}&name={urllib.parse.quote_plus(platform_name)}"
        resp = requests.get(endpoint, headers=self._headers, proxies=self._proxies)

        resp.raise_for_status()
        platform_resp = resp.json()

        if "platforms" not in platform_resp:
            raise HTTPError(
                f"platforms '{platform_name}' not found for provider '{provider_name}'"
            )

        platforms = platform_resp["platforms"]

        if len(platforms) == 0:
            raise HTTPError(f"Empty platform list for provider '{provider_name}'")

        platform = platforms[0]
        platform["specs"] = json.loads(platform.get("metadata", {}))

        return platform

    def create_session(
        self,
        deduplication_id: str,
        max_duration_s: int,
        max_idle_duration_s: int,
    ) -> str:
        """Start a session, when started job can be sent and executed on it

        :param provider_name: the quantum provider to target, quandela by default
        :param platform_name: the platform to built the session from
        :param deduplcation_id: the session creation will return any running session with that id
        :param max_duration_s: max duration (in second) before the session is terminated
        :param max_idle_duration_s: max duration (in second) before the session is terminated
        """

        if self._session_id:
            raise Exception("A session is already attached to this RPC handler")

        payload = {
            "project_id": self._project_id,
            "platform_id": self._platform_id,
            "deduplication_id": deduplication_id,
            "max_duration": f"{max_duration_s}s",
            "max_idle_duration": f"{max_idle_duration_s}s",
            "name": f"pcvl-{datetime.now():%Y-%m-%d-%H-%M-%S}",
        }

        endpoint = f"{self._url}{_ENDPOINT_SESSION}"
        request = requests.post(
            endpoint, headers=self._headers, json=payload, proxies=self._proxies
        )

        try:
            request.raise_for_status()
            session = request.json()

            self._session_id = session["id"]
            get_logger().info("Start Scaleway Session", channel.general)
        except Exception:
            raise HTTPError(request.json())

        return self._session_id

    def terminate_session(self) -> None:
        """Stop the attached session, all jobs will be cancelled and still accessible"""

        if not self._session_id:
            raise Exception("Cannot terminate session because session_id is None")

        endpoint = f"{self._url}{_ENDPOINT_SESSION}/{self._session_id}/terminate"
        request = requests.post(endpoint, headers=self._headers, proxies=self._proxies)

        request.raise_for_status()

    def delete_session(self) -> None:
        """Stop and delete the attached session, all jobs will be deleted"""

        if not self._session_id:
            raise Exception("Cannot delete session because session_id is None")

        endpoint = f"{self._url}{_ENDPOINT_SESSION}/{self._session_id}"
        request = requests.delete(
            endpoint, headers=self._headers, proxies=self._proxies
        )

        request.raise_for_status()

    def create_job(self, payload: dict) -> str:
        """Create and start on new job on the attached session

        :param payload: the perceval circuit and run parameters to be executed on the attached session
        """
        if not self._session_id:
            raise Exception("Cannot create job because session_id is None")

        scw_payload = {
            "name": payload.get("job_name"),
            "circuit": {"percevalCircuit": json.dumps(payload.get("payload", {}))},
            "project_id": self._project_id,
            "session_id": self._session_id,
        }

        endpoint = f"{self._url}{_ENDPOINT_JOB}"
        request = requests.post(
            endpoint, headers=self._headers, json=scw_payload, proxies=self._proxies
        )

        try:
            request.raise_for_status()
            job = request.json()

            self.instance_id = job["id"]
        except Exception:
            raise HTTPError(request.json())

        return job["id"]

    def cancel_job(self, job_id: str) -> None:
        """Cancel the running job

        :param job_id: job id to cancel
        """
        endpoint = f"{self.__build_endpoint(_ENDPOINT_JOB)}/{job_id}/cancel"
        request = requests.post(endpoint, headers=self._headers, proxies=self._proxies)
        request.raise_for_status()

    def rerun_job(self, job_id: str) -> str:
        """Rerun a job

        :param job_id: job id to rerun
        :return: new job id
        """
        raise NotImplementedError(
            "rerun_job method is not implemented for Scaleway RPCHandler"
        )

    def get_job_status(self, job_id: str) -> dict:
        """Fetch the current job status

        :param job_id: job id to fetch for
        :return: dictionary of the current job status
        """
        endpoint = f"{self.__build_endpoint(_ENDPOINT_JOB)}/{job_id}"

        # requests may throw an IO Exception, let the user deal with it
        resp = requests.get(endpoint, headers=self._headers, proxies=self._proxies)
        resp.raise_for_status()

        job = resp.json()

        created_at = self.__to_date(job.get("created_at"))
        started_at = self.__to_date(job.get("started_at"))
        duration = self.__get_duration(started_at)
        status = job.get("status")

        if status == "cancelled":
            status = "canceled"
        elif status == "cancelling":
            status = "cancel_requested"

        return {
            "creation_datetime": created_at,
            "duration": duration,
            "failure_code": None,
            "last_intermediate_results": None,
            "msg": "ok",
            "progress": 1 if status == "completed" else 0,
            "progress_message": job.get("progress_message"),
            "start_time": started_at,
            "status": status,
            "status_message": job.get("progress_message"),
        }

    def get_job_results(self, job_id: str) -> dict:
        """Get result of the job execution

        :param job_id: job id to fetch for
        :return: dictionary of the results
        """
        endpoint = f"{self.__build_endpoint(_ENDPOINT_JOB)}/{job_id}/results"

        # requests may throw an IO Exception, let the user deal with it
        resp = requests.get(endpoint, headers=self._headers, proxies=self._proxies)
        resp.raise_for_status()

        resp_dict = resp.json()

        result_payload = None
        duration = None
        results = resp_dict.get("job_results", [])

        if len(results) > 0:
            first_result = results[0]

            duration = self.__get_duration(
                self.__to_date(first_result.get("created_at"))
            )

            result = first_result.get("result", None)

            if result is None or result == "":
                url = first_result.get("url", None)

                if url is not None:
                    resp = requests.get(url, proxies=self._proxies)
                    resp.raise_for_status()

                    result_payload = resp.text
                else:
                    raise Exception("Got result with empty 'result' and 'url' fields")
            else:
                result_payload = result

        return {
            "duration": duration,
            "intermediate_results": [],
            "job_id": resp_dict.get("job_id"),
            "results": result_payload,
            "results_type": None,
        }

    def __build_endpoint(self, endpoint) -> str:
        return f"{self._url}{endpoint}"

    def __to_date(self, date: str) -> float | None:
        if not date:
            return None

        # Compat for python 3.10
        if date.endswith("Z"):
            date = date[:-1]

        return datetime.fromisoformat(date).timestamp()

    def __get_duration(self, start_time: float | None) -> int | None:
        return (
            timedelta(seconds=time.time() - start_time).seconds if start_time else None
        )
