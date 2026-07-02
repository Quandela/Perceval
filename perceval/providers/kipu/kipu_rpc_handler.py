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

from datetime import datetime

_MISSING_QHUB_MSG = (
    "The Kipu Quantum Hub provider requires the 'qhub-api' package. "
    "Install it with: pip install perceval[kipu]"
)

_SIM_BELENOS = "quandela.sim.belenos"
_QPU_BELENOS = "quandela.qpu.belenos"

_SUPPORTED_BACKENDS = (_SIM_BELENOS, _QPU_BELENOS)

_ALIASES = {
    "sim:belenos": _SIM_BELENOS,
    "qpu:belenos": _QPU_BELENOS,
}

_STATUS_MAP = {
    "PENDING": "waiting",
    "RUNNING": "running",
    "COMPLETED": "completed",
    "FAILED": "error",
    "CANCELLING": "cancel_requested",
    "CANCELLED": "canceled",
    "ABORTED": "error",
    "UNKNOWN": "unknown",
}

_BACKEND_TYPE_MAP = {
    "SIMULATOR": "simulator",
    "QPU": "qpu",
    "ANNEALER": "qpu",
    "UNKNOWN": "simulator",
}


def _resolve_backend_id(platform_name: str) -> str:
    """Resolve a Perceval platform_name to a Kipu Hub backend id.

    Accepts either a raw Hub backend id (e.g. "quandela.sim.belenos") or a
    friendly alias (e.g. "sim:belenos").

    :param platform_name: the platform name passed to the Kipu Session
    :return: a supported Hub backend id
    :raises ValueError: if the name resolves to an unsupported backend
    """
    backend_id = _ALIASES.get(platform_name, platform_name)
    if backend_id not in _SUPPORTED_BACKENDS:
        raise ValueError(
            f"Unknown Kipu platform '{platform_name}'. "
            f"Supported backends: {', '.join(_SUPPORTED_BACKENDS)} "
            f"(aliases: {', '.join(_ALIASES)})."
        )
    return backend_id


def _to_perceval_status(hub_status: str | None) -> str:
    """Map a Hub JobStatus value to a Perceval status string.

    A missing status (None) is not a map key, so it falls back to "unknown".
    """
    return _STATUS_MAP.get(hub_status, "unknown")


def _import_qhub() -> dict:
    """Lazily import qhub-api symbols. Raises a clear ImportError if absent."""
    try:
        from qhub.api.quantum import HubQuantumClient
        from qhub.api.credentials import DefaultCredentialsProvider
        from qhub.api.quantum.jobs import (
            CreateJobRequestInput_QuandelaSimBelenos,
            CreateJobRequestInput_QuandelaQpuBelenos,
            CreateJobRequestInputParams_QuandelaSimBelenos,
            CreateJobRequestInputParams_QuandelaQpuBelenos,
        )
    except ImportError as e:
        raise ImportError(_MISSING_QHUB_MSG) from e
    return {
        "client": HubQuantumClient,
        "credentials": DefaultCredentialsProvider,
        "input": {
            _SIM_BELENOS: CreateJobRequestInput_QuandelaSimBelenos,
            _QPU_BELENOS: CreateJobRequestInput_QuandelaQpuBelenos,
        },
        "params": {
            _SIM_BELENOS: CreateJobRequestInputParams_QuandelaSimBelenos,
            _QPU_BELENOS: CreateJobRequestInputParams_QuandelaQpuBelenos,
        },
    }


def _build_httpx_client(proxies: dict | None):
    """Build an httpx.Client routing through `proxies`, or None if none set.

    Perceval proxies are requests-style ({scheme: url}); httpx wants per-scheme
    transport mounts. httpx is imported lazily — it ships with qhub-api, not base
    Perceval.
    """
    if not proxies:
        return None
    import httpx
    mounts = {
        f"{scheme}://": httpx.HTTPTransport(proxy=url)
        for scheme, url in proxies.items()
    }
    # explicit timeout: injecting httpx_client bypasses qhub-api's 60s default
    return httpx.Client(timeout=60, mounts=mounts)


def _to_timestamp(value):
    """Convert an ISO-8601 datetime string to a POSIX timestamp (float).

    Returns 0. for falsy input — Perceval's status parser coerces the timing
    fields with float()/int(), so a numeric default avoids a spurious error log
    on every poll (see ``_retrieve_from_response`` in remote_job.py). Tolerates a
    trailing 'Z' (Python 3.10's ``datetime.fromisoformat`` does not accept it).
    """
    if not value:
        return 0.
    if isinstance(value, str) and value.endswith("Z"):
        # explicit offset: 3.10's fromisoformat rejects 'Z'; naive would be local
        value = value[:-1] + "+00:00"
    return datetime.fromisoformat(value).timestamp()


class KipuRPCHandler:
    """Duck-typed RPC handler for the Kipu Quantum Hub, backed by qhub-api.

    :param platform_name: alias or Hub backend id (e.g. "quandela.sim.belenos")
    :param url: optional Hub base URL; when None the qhub-api client uses its
        own default Hub endpoint
    :param token: optional Kipu Personal Access Token (PAT); when None it is
        resolved from the environment or the `qhubctl login` config file
    :param organization_id: optional Kipu organization id; when None your
        personal account is used
    :param proxies: optional protocol->URL proxy mapping (requests-style, e.g.
        ``{"https": "http://proxy:8080"}``); routed through the underlying httpx
        client
    :param client: optional pre-built HubQuantumClient (used for testing)
    """

    def __init__(self, platform_name, url=None, token=None, organization_id=None,
                 proxies=None, client=None):
        self._platform_name = platform_name
        self._url = url
        self._token = token
        self._organization_id = organization_id
        self._proxies = proxies or {}
        self._backend_id = _resolve_backend_id(platform_name)
        self._client = client if client is not None else self._build_client()

    def _build_client(self):
        qhub = _import_qhub()
        # explicit token used as-is; None -> resolved from env / `qhubctl login`
        api_key = qhub["credentials"](self._token).get_access_token()
        kwargs = {
            "api_key": api_key,
            "organization_id": self._organization_id,
        }
        if self._url:
            kwargs["base_url"] = self._url
        httpx_client = _build_httpx_client(self._proxies)
        if httpx_client is not None:
            kwargs["httpx_client"] = httpx_client
        return qhub["client"](**kwargs)

    @property
    def name(self) -> str:
        return self._platform_name

    @property
    def url(self) -> str:
        return self._url

    @property
    def proxies(self) -> dict:
        return self._proxies

    @property
    def headers(self) -> dict:
        # qhub-api manages auth via api_key; no raw headers to expose. Defined
        # because RemoteJob._to_dict() reads it (job serialization / rerun).
        return {}

    def create_job(self, payload: dict) -> str:
        """Submit a Perceval job payload to the Hub. Returns the Hub job id."""
        qhub = _import_qhub()
        inner = payload.get("payload") or {}
        shots = inner.get("max_shots") or inner.get("max_samples") or 0

        input_cls = qhub["input"][self._backend_id]
        params_cls = qhub["params"][self._backend_id]

        job = self._client.jobs.create_job(
            backend_id=self._backend_id,
            shots=shots,
            input=input_cls(value=inner),
            input_format="PERCEVAL",
            input_params=params_cls(
                pcvl_version=payload.get("pcvl_version"),
                process_id=payload.get("process_id"),
            ),
            sdk_provider="PERCEVAL",
            name=payload.get("job_name"),
        )
        return job.id

    def get_job_status(self, job_id: str) -> dict:
        """Map the Hub job status to Perceval's status dict.

        ``get_job_status`` returns the live status; ``get_job`` adds the timing
        fields (creation/start/duration). Progress and failure code are not
        exposed by the Hub; the error message for a failed job comes from the
        result payload via ``_fetch_error_message`` (surfaced as status_message).
        """
        status = _to_perceval_status(getattr(self._client.jobs.get_job_status(job_id), "status", None))
        job = self._client.jobs.get_job(job_id)
        status_message = self._fetch_error_message(job_id) if status == "error" else None
        return {
            "creation_datetime": _to_timestamp(getattr(job, "created_at", None)),
            "duration": getattr(job, "runtime", None) or 0.,
            "failure_code": None,
            "last_intermediate_results": None,
            "msg": "ok",
            "progress": 1.0 if status == "completed" else 0.0,
            "progress_message": None,
            "start_time": _to_timestamp(getattr(job, "started_at", None)),
            "status": status,
            "status_message": status_message,
        }

    def _fetch_error_message(self, job_id: str):
        """Best-effort error string for a failed job.

        A failed Quandela job stores its error message (not a serialized result)
        in the result payload's ``results`` field. Returns it as a string, or
        None if it cannot be retrieved — status reporting must never break on the
        error lookup.
        """
        try:
            result = self._client.jobs.get_job_result(job_id)
        except Exception:
            return None
        message = result.get("results") if isinstance(result, dict) else None
        return message if isinstance(message, str) else None

    def get_job_results(self, job_id: str) -> dict:
        """Fetch Hub job results, normalized to Perceval's result envelope.

        The Hub stores Quandela-native results already in Perceval's shape.
        """
        result = self._client.jobs.get_job_result(job_id)
        return {
            "duration": result.get("duration"),
            "intermediate_results": result.get("intermediate_results", []),
            "job_id": result.get("job_id", job_id),
            "results": result.get("results"),
            "results_type": result.get("results_type"),
        }

    def cancel_job(self, job_id: str) -> None:
        """Cancel a running Hub job."""
        self._client.jobs.cancel_job(job_id)

    def rerun_job(self, job_id: str) -> str:
        """Not supported by the Kipu Quantum Hub provider; always raises NotImplementedError."""
        raise NotImplementedError(
            "rerun_job is not implemented for the Kipu Quantum Hub provider"
        )

    def fetch_platform_details(self) -> dict:
        """Return the Hub's platform details for this backend.

        The Hub's backend-config endpoint relays Quandela Cloud's native
        platform payload, which already matches the shape Perceval consumes
        (a ``specs`` block with ``architecture``/``specific_circuit``,
        ``available_commands``, ``constraints``, ``detector``, ... plus
        top-level ``perfs`` and ``status``). We forward it as-is, only
        ensuring ``type`` is set: the payload omits it, and Perceval would
        otherwise default every platform to "simulator".
        """
        details = self._client.backends.get_backend_config(self._backend_id)
        hub_type = "SIMULATOR" if ".sim." in self._backend_id else "QPU"
        details.setdefault("type", _BACKEND_TYPE_MAP.get(hub_type, "simulator"))
        return details
