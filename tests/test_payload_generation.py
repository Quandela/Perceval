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
from _mock_rpc_handler import MockRPCHandler
from perceval import RemoteProcessor, Circuit, BasicState, PostSelect
from perceval.serialization._constants import ZIP_PREFIX, BS_TAG, SEP, PCVL_PREFIX, POSTSELECT_TAG


COMMAND_NAME = 'my_command'

def _get_remote_processor():
    rp = RemoteProcessor(rpc_handler=MockRPCHandler())
    rp.set_circuit(Circuit(8))
    return rp


def test_payload_basics():
    rp = _get_remote_processor()
    data = rp.prepare_job_payload(COMMAND_NAME)
    assert 'platform_name' in data and data['platform_name'] == MockRPCHandler.name
    assert 'pcvl_version' in data
    assert 'process_id' in data
    assert 'payload' in data

    payload = data['payload']
    assert 'command' in payload and payload['command'] == COMMAND_NAME
    assert 'circuit' in payload and payload['circuit'].startswith(ZIP_PREFIX)  # Circuits are compressed in payloads
    assert 'input_state' not in payload  # No input state was passed

    input_state = BasicState([1, 0]*4)
    rp.with_input(input_state)
    new_payload = rp.prepare_job_payload(COMMAND_NAME)['payload']
    assert ('input_state' in new_payload and
        new_payload['input_state'] == f"{PCVL_PREFIX}{BS_TAG}{SEP}{str(input_state)}")


def test_payload_parameters():
    n_params = 5
    rp = _get_remote_processor()
    params = {f'param{i}': f'value{i}' for i in range(n_params)}
    rp.set_parameters(params)
    with pytest.warns(DeprecationWarning):
        rp.set_parameter('g2', 0.05)
    payload = rp.prepare_job_payload(COMMAND_NAME)['payload']
    assert 'parameters' in payload
    for i in range(n_params):
        assert f'param{i}' in payload['parameters']
        assert payload['parameters'][f'param{i}'] == f'value{i}'


def test_payload_heralds():
    rp = _get_remote_processor()
    payload = rp.prepare_job_payload(COMMAND_NAME)['payload']
    assert 'heralds' not in payload

    rp.add_herald(3, 1)
    rp.add_herald(4, 0)
    payload = rp.prepare_job_payload(COMMAND_NAME)['payload']
    assert 'heralds' in payload
    assert payload['heralds'][3] == 1
    assert payload['heralds'][4] == 0


def test_payload_postselect():
    rp = _get_remote_processor()
    payload = rp.prepare_job_payload(COMMAND_NAME)['payload']
    assert 'postselect' not in payload

    rp.set_postselection(PostSelect('[3]==1 & [4]==0'))
    payload = rp.prepare_job_payload(COMMAND_NAME)['payload']
    assert 'postselect' in payload
    assert payload['postselect'].startswith(f"{PCVL_PREFIX}{POSTSELECT_TAG}{SEP}")
