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

import pytest

from perceval.providers.kipu.kipu_rpc_handler import (
    _resolve_backend_id,
    _to_perceval_status,
)


def test_resolve_backend_id_passthrough():
    assert _resolve_backend_id("quandela.sim.belenos") == "quandela.sim.belenos"
    assert _resolve_backend_id("quandela.qpu.belenos") == "quandela.qpu.belenos"


def test_resolve_backend_id_alias():
    assert _resolve_backend_id("sim:belenos") == "quandela.sim.belenos"
    assert _resolve_backend_id("qpu:belenos") == "quandela.qpu.belenos"


def test_resolve_backend_id_unknown_raises():
    with pytest.raises(ValueError, match="quandela.sim.belenos"):
        _resolve_backend_id("nonsense")


@pytest.mark.parametrize("hub_status,expected", [
    ("PENDING", "waiting"),
    ("RUNNING", "running"),
    ("COMPLETED", "completed"),
    ("FAILED", "error"),
    ("CANCELLING", "cancel_requested"),
    ("CANCELLED", "canceled"),
    ("ABORTED", "error"),
    ("UNKNOWN", "unknown"),
    (None, "unknown"),
    ("SOMETHING_NEW", "unknown"),
])
def test_to_perceval_status(hub_status, expected):
    assert _to_perceval_status(hub_status) == expected
