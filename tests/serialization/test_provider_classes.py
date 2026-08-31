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
import pytest

from perceval import ContextManager, ScalewayConfig, KipuConfig
from perceval.providers.kipu.kipu_communication_layer import KipuCommunicationLayer
from perceval.providers.kipu.kipu_rpc_handler import KipuRPCHandler
from perceval.providers.quandela.quandela_communication_layer import QuandelaCommunicationLayer
from perceval.providers.quandela.rpc_handler import RPCHandler as QuandelaRPCHandler
from perceval.providers.scaleway.scaleway_communication_layer import ScalewayCommunicationLayer
from perceval.providers.scaleway.scaleway_rpc_handler import RPCHandler as ScalewayRPCHandler
from perceval.providers import AbstractRemoteConfig, RemoteConfig
from perceval.serialization import InputArchive, OutputArchive, Serialization


_PLATFORM_DETAILS = {
    "id": "platform-id",
    "status": "available",
    "specs": {"available_commands": ["probs"]},
    "type": "simulator",
    "perfs": {"transmittance": .75},
}


def config_manager(config: AbstractRemoteConfig, token: str, proxies = None):
    initial_token = config.get_token()
    initial_proxies = config.get_proxies()

    def set_values(t, p):
        config.set_token(t)
        config.set_proxies(p)
        config.save()

    return ContextManager(lambda: set_values(token, proxies),
                          lambda: set_values(initial_token, initial_proxies))


def _round_trip(obj):
    archive = OutputArchive()
    Serialization.serialize(obj, archive)
    return Serialization.deserialize(InputArchive.from_text(archive.to_text()))


def test_quandela_rpc_handler_serialization():
    handler = QuandelaRPCHandler(
        "sim:test", "https://cloud.example", "token", {"https": "proxy"}
    )
    handler.request_timeout = 42

    with config_manager(RemoteConfig(), ""):
        with pytest.raises(ConnectionError):  # Nothing set in the remote config
            restored = _round_trip(handler)

    with config_manager(RemoteConfig(), "token", {"https": "proxy"}):
        restored = _round_trip(handler)

    assert type(restored) is QuandelaRPCHandler
    assert restored.__dict__ == handler.__dict__


def test_scaleway_rpc_handler_serialization(monkeypatch):
    monkeypatch.setattr(ScalewayRPCHandler, "get_platform", lambda *_, **__: _PLATFORM_DETAILS)
    handler = ScalewayRPCHandler.__new__(ScalewayRPCHandler)  # We need to avoid the automatic use of get_platform()
    handler._project_id = "project-id"
    handler._url = "https://api.example"
    handler._proxies = {"https": "proxy"}
    handler._session_id = "session-id"
    handler._platform_name = "sim:test"
    handler._headers = {"X-Auth-Token": "token"}
    handler._provider_name = "quandela"
    handler._platform_id = _PLATFORM_DETAILS["id"]

    with pytest.raises(ConnectionError):  # Nothing set in the remote config
        restored = _round_trip(handler)

    with config_manager(ScalewayConfig(), "token", {"https": "proxy"}):
        restored = _round_trip(handler)

    assert type(restored) is ScalewayRPCHandler
    assert restored.__dict__ == handler.__dict__


def test_kipu_rpc_handler_serialization(monkeypatch):
    rebuilt_client = object()
    monkeypatch.setattr(KipuRPCHandler, "_build_client", lambda _: rebuilt_client)
    handler = KipuRPCHandler(
        "quandela.sim.belenos",
        url="https://hub.example",
        token="token",
        organization_id="organization-id",
        proxies={"https": "proxy"},
        client=object(),
    )

    # The KipuRPCHandler can live without token - no need to test without token
    with config_manager(KipuConfig(), "token", {"https": "proxy"}):
        restored = _round_trip(handler)

    assert restored._client is rebuilt_client
    assert restored._platform_name == handler._platform_name
    assert restored._url == handler._url
    assert restored._token == handler._token
    assert restored._organization_id == handler._organization_id
    assert restored._proxies == handler._proxies
    assert restored._backend_id == handler._backend_id


def test_quandela_communication_layer_serialization(monkeypatch):
    monkeypatch.setattr(
        QuandelaRPCHandler, "fetch_platform_details", lambda _: _PLATFORM_DETAILS
    )
    layer = QuandelaCommunicationLayer(
        "sim:test", "token", "https://cloud.example"
    )
    layer._status = "unavailable"
    layer._specs.clear()

    with config_manager(RemoteConfig(), "token"):
        restored = _round_trip(layer)

    assert type(restored) is QuandelaCommunicationLayer
    assert type(restored._rpc_handler) is QuandelaRPCHandler
    assert restored._status == "available"
    assert restored._specs.available_commands == ["probs"]


def test_scaleway_communication_layer_serialization(monkeypatch):
    monkeypatch.setattr(
        ScalewayRPCHandler, "get_platform", lambda *_args, **_kwargs: _PLATFORM_DETAILS
    )
    layer = ScalewayCommunicationLayer(
        "sim:test",
        "project-id",
        "token",
        max_idle_duration_s=10,
        max_duration_s=20,
        deduplication_id="deduplication-id",
    )
    layer._status = "unavailable"
    layer._specs.clear()

    with config_manager(ScalewayConfig(), "token"):
        restored = _round_trip(layer)

    assert type(restored) is ScalewayCommunicationLayer
    assert type(restored._rpc_handler) is ScalewayRPCHandler
    assert restored._status == "available"
    assert restored._specs.available_commands == ["probs"]
    assert restored._deduplication_id == "deduplication-id"
    assert restored._max_idle_duration_s == 10
    assert restored._max_duration_s == 20


def test_kipu_communication_layer_serialization(monkeypatch):
    clients = []

    def build_client(_):
        client = object()
        clients.append(client)
        return client

    monkeypatch.setattr(KipuRPCHandler, "_build_client", build_client)
    monkeypatch.setattr(
        KipuRPCHandler, "fetch_platform_details", lambda _: _PLATFORM_DETAILS
    )
    layer = KipuCommunicationLayer(
        "quandela.sim.belenos", token="token", organization_id="organization-id"
    )
    layer._status = "unavailable"
    layer._specs.clear()

    with config_manager(KipuConfig(), "token"):
        restored = _round_trip(layer)

    assert type(restored) is KipuCommunicationLayer
    assert type(restored._rpc_handler) is KipuRPCHandler
    assert restored._rpc_handler._client is not clients[0]
    assert restored._rpc_handler._client is clients[1]
    assert restored._status == "available"
    assert restored._specs.available_commands == ["probs"]
