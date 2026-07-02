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

import socket
import sys
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock
import json

import pytest
# qhub is a namespace package; guard on the concrete submodule so the suite skips when [kipu] is absent.
pytest.importorskip("qhub.api.quantum")

import perceval.providers.kipu.kipu_rpc_handler as kipu_mod
from perceval.providers.kipu.kipu_rpc_handler import KipuRPCHandler


def _make_handler(client=None, platform_name="quandela.sim.belenos", url=None):
    return KipuRPCHandler(
        platform_name=platform_name,
        url=url,
        token="test-token",
        organization_id="test-org",
        client=client or MagicMock(),
    )


def test_handler_properties():
    handler = _make_handler()
    assert handler.name == "quandela.sim.belenos"
    assert handler.url is None
    assert handler.proxies == {}
    assert handler.headers == {}


def test_build_httpx_client_none_without_proxies():
    assert kipu_mod._build_httpx_client({}) is None
    assert kipu_mod._build_httpx_client(None) is None


def test_build_httpx_client_routes_traffic_through_proxy():
    # A one-shot local socket standing in for a proxy: an HTTP/1.1 request routed
    # through a proxy uses absolute-form ("GET http://host/path"), whereas a direct
    # request uses origin-form ("GET /path"). Seeing absolute-form proves routing.
    captured = {}

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]

    def serve():
        conn, _ = server.accept()
        with conn:
            data = b""
            while b"\r\n\r\n" not in data:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                data += chunk
            captured["request_line"] = data.split(b"\r\n", 1)[0].decode()
            conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok")

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()

    client = kipu_mod._build_httpx_client({"http": f"http://127.0.0.1:{port}"})
    try:
        resp = client.get("http://proxy-target.test/ping")
    finally:
        client.close()
        thread.join(timeout=5)
        server.close()

    assert resp.status_code == 200
    assert captured["request_line"].startswith("GET http://proxy-target.test/ping")


@pytest.mark.parametrize("proxies, injected", [
    ({"https": "http://proxy:8080"}, True),
    (None, False),
])
def test_build_client_injects_httpx_client_only_when_proxied(monkeypatch, proxies, injected):
    captured = {}
    monkeypatch.setattr(kipu_mod, "_import_qhub", lambda: {
        "client": lambda **kwargs: captured.update(kwargs) or MagicMock(),
        "credentials": lambda token: SimpleNamespace(get_access_token=lambda: "key"),
    })
    KipuRPCHandler(platform_name="quandela.sim.belenos", token="test-token", proxies=proxies)
    assert ("httpx_client" in captured) is injected


def test_handler_url_passthrough():
    handler = _make_handler(url="https://hub.example.test/quantum")
    assert handler.url == "https://hub.example.test/quantum"


def test_handler_resolves_backend_id():
    handler = _make_handler(platform_name="sim:belenos")
    assert handler._backend_id == "quandela.sim.belenos"


def test_missing_qhub_raises_clear_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "qhub.api.quantum", None)
    with pytest.raises(ImportError, match=r"pip install perceval\[kipu\]"):
        kipu_mod._import_qhub()


def test_create_job_builds_typed_request_and_returns_id():
    client = MagicMock()
    client.jobs.create_job.return_value = SimpleNamespace(id="job-123")
    handler = _make_handler(client=client)

    payload = {
        "pcvl_version": "1.0",
        "process_id": "proc-1",
        "platform_name": "quandela.sim.belenos",
        "job_name": "toy",
        "payload": {"command": "samples", "max_shots": 1000},
    }
    job_id = handler.create_job(payload)

    assert job_id == "job-123"
    kwargs = client.jobs.create_job.call_args.kwargs
    assert kwargs["backend_id"] == "quandela.sim.belenos"
    assert kwargs["shots"] == 1000
    assert kwargs["input_format"] == "PERCEVAL"
    assert kwargs["sdk_provider"] == "PERCEVAL"
    assert kwargs["name"] == "toy"
    assert kwargs["input"].value == payload["payload"]
    assert kwargs["input_params"].pcvl_version == "1.0"
    assert kwargs["input_params"].process_id == "proc-1"


def test_create_job_falls_back_to_max_samples_for_shots():
    client = MagicMock()
    client.jobs.create_job.return_value = SimpleNamespace(id="job-9")
    handler = _make_handler(client=client)
    handler.create_job({"payload": {"command": "samples", "max_samples": 500}})
    assert client.jobs.create_job.call_args.kwargs["shots"] == 500


def test_get_job_status_maps_completed_with_timing():
    client = MagicMock()
    client.jobs.get_job_status.return_value = SimpleNamespace(status="COMPLETED")
    client.jobs.get_job.return_value = SimpleNamespace(
        created_at="2024-03-01T15:33:43Z",
        started_at="2024-03-01T15:33:45Z",
        runtime=12,
    )
    handler = _make_handler(client=client)

    status = handler.get_job_status("job-1")
    client.jobs.get_job_status.assert_called_once_with("job-1")
    client.jobs.get_job.assert_called_once_with("job-1")
    assert status["status"] == "completed"
    assert status["progress"] == pytest.approx(1.0)
    assert status["msg"] == "ok"
    assert status["duration"] == 12
    assert isinstance(status["creation_datetime"], float)
    assert isinstance(status["start_time"], float)
    assert status["start_time"] - status["creation_datetime"] == pytest.approx(2.0)
    for key in ("creation_datetime", "duration", "start_time",
                "progress_message", "status_message", "failure_code"):
        assert key in status
    assert status["status_message"] is None
    client.jobs.get_job_result.assert_not_called()


def test_get_job_status_maps_running():
    client = MagicMock()
    client.jobs.get_job_status.return_value = SimpleNamespace(status="RUNNING")
    client.jobs.get_job.return_value = SimpleNamespace(
        created_at=None, started_at=None, runtime=None)
    handler = _make_handler(client=client)
    status = handler.get_job_status("job-2")
    assert status["status"] == "running"
    assert status["progress"] == pytest.approx(0.0)
    # missing timing fields default to a number, not None: Perceval coerces
    # them with float()/int(), so None would spam an error log on every poll
    assert status["start_time"] == pytest.approx(0.0)
    assert status["duration"] == pytest.approx(0.0)
    client.jobs.get_job_result.assert_not_called()


def test_get_job_status_failed_surfaces_error_message():
    client = MagicMock()
    client.jobs.get_job_status.return_value = SimpleNamespace(status="FAILED")
    client.jobs.get_job.return_value = SimpleNamespace(
        created_at=None, started_at=None, runtime=None)
    client.jobs.get_job_result.return_value = {
        "results": "Error -3 while decompressing data: incorrect data check"}
    handler = _make_handler(client=client)

    status = handler.get_job_status("job-err")
    assert status["status"] == "error"
    client.jobs.get_job_result.assert_called_once_with("job-err")
    assert status["status_message"] == (
        "Error -3 while decompressing data: incorrect data check")


def test_get_job_status_failed_error_lookup_is_best_effort():
    client = MagicMock()
    client.jobs.get_job_status.return_value = SimpleNamespace(status="FAILED")
    client.jobs.get_job.return_value = SimpleNamespace(
        created_at=None, started_at=None, runtime=None)
    client.jobs.get_job_result.side_effect = RuntimeError("boom")
    handler = _make_handler(client=client)
    status = handler.get_job_status("job-err")
    assert status["status"] == "error"
    assert status["status_message"] is None


def test_get_job_results_passthrough_envelope():
    inner = json.dumps({"results": ":PCVL:BasicState:|0,1>", "logical_perf": 1})
    client = MagicMock()
    client.jobs.get_job_result.return_value = {
        "duration": 2,
        "intermediate_results": [],
        "job_id": "job-7",
        "results": inner,
        "results_type": None,
        "shots": 1000,
    }
    handler = _make_handler(client=client)

    out = handler.get_job_results("job-7")
    client.jobs.get_job_result.assert_called_once_with("job-7")
    assert set(out) == {"duration", "intermediate_results", "job_id",
                        "results", "results_type"}
    assert out["results"] == inner
    assert out["job_id"] == "job-7"
    assert out["duration"] == 2


def test_get_job_results_defaults_job_id_when_absent():
    client = MagicMock()
    client.jobs.get_job_result.return_value = {"results": None}
    handler = _make_handler(client=client)
    out = handler.get_job_results("fallback-id")
    assert out["job_id"] == "fallback-id"
    assert out["intermediate_results"] == []


def test_cancel_job_calls_client():
    client = MagicMock()
    handler = _make_handler(client=client)
    handler.cancel_job("job-x")
    client.jobs.cancel_job.assert_called_once_with("job-x")


def test_rerun_job_not_implemented():
    handler = _make_handler()
    with pytest.raises(NotImplementedError):
        handler.rerun_job("job-x")


# Trimmed sample of the Hub backend-config payload (Quandela-native).
_CONFIG_PAYLOAD = {
    "name": "sim:belenos",
    "status": "online",
    "perfs": {},
    "specs": {
        "available_commands": ["sample_count", "samples"],
        "connected_input_modes": [1, 3, 5, 7],
        "constraints": {
            "max_mode_count": 20,
            "min_mode_count": 1,
            "max_photon_count": 10,
            "min_photon_count": 1,
        },
        "detector": "threshold",
        "specific_circuit": ":PCVL:zip:fake",
    },
}


def test_fetch_platform_details_forwards_config_and_injects_type():
    client = MagicMock()
    client.backends.get_backend_config.return_value = dict(_CONFIG_PAYLOAD)
    handler = _make_handler(client=client)

    details = handler.fetch_platform_details()
    client.backends.get_backend_config.assert_called_once_with(
        "quandela.sim.belenos")
    assert details["specs"] == _CONFIG_PAYLOAD["specs"]
    assert details["perfs"] == {}
    assert details["status"] == "online"
    assert details["type"] == "simulator"


def test_fetch_platform_details_injects_qpu_type():
    client = MagicMock()
    client.backends.get_backend_config.return_value = dict(_CONFIG_PAYLOAD)
    handler = _make_handler(client=client, platform_name="quandela.qpu.belenos")
    details = handler.fetch_platform_details()
    assert details["type"] == "qpu"


def test_fetch_platform_details_does_not_override_existing_type():
    payload = dict(_CONFIG_PAYLOAD, type="qpu")
    client = MagicMock()
    client.backends.get_backend_config.return_value = payload
    handler = _make_handler(client=client)  # sim backend ...
    details = handler.fetch_platform_details()
    assert details["type"] == "qpu"  # ... but payload's own type wins
