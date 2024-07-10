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
from perceval import RemoteProcessor, Circuit, BasicState, PostSelect, catalog
from perceval.serialization._constants import ZIP_PREFIX, BS_TAG, SEP, PCVL_PREFIX, POSTSELECT_TAG
from perceval.serialization import deserialize


COMMAND_NAME = 'my_command'

def _get_remote_processor(m: int = 8):
    return RemoteProcessor(rpc_handler=MockRPCHandler(), m=m)


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


def test_payload_min_detected_photons():
    rp = _get_remote_processor()
    payload = rp.prepare_job_payload(COMMAND_NAME)['payload']
    assert 'parameters' not in payload

    rp.min_detected_photons_filter(2)
    payload = rp.prepare_job_payload(COMMAND_NAME)['payload']
    assert 'min_detected_photons' in payload['parameters']
    assert payload['parameters']['min_detected_photons'] == 2


def test_payload_cnot():
    rp = _get_remote_processor()
    heralded_cnot = catalog['heralded cnot'].build_processor()
    pp_cnot = catalog['postprocessed cnot'].build_processor()
    rp.add(0, heralded_cnot)
    rp.add(0, pp_cnot)
    rp.add(4, pp_cnot)
    assert rp.m == 8
    assert rp.circuit_size == 14  # 8 modes of interest + 6 ancillaries

    input_state = BasicState([1, 0]*4)
    rp.with_input(input_state)

    payload = rp.prepare_job_payload(COMMAND_NAME)['payload']
    assert 'input_state' in payload
    assert payload['input_state'] == f"{PCVL_PREFIX}{BS_TAG}{SEP}{str(input_state*BasicState('|1,1,0,0,0,0>'))}"

    # Heralds come from the 3 CNOT gates
    assert 'heralds' in payload and len(payload['heralds']) == 6
    # Heralds from the heralded CNOT are put after the modes of interest (0 to 7) and 1 photon is injected in each
    assert payload['heralds'][8] == 1 and payload['heralds'][9] == 1
    # Herlads from the postprocessed CNOT come last (because it was added last) and no photon is required ...
    assert payload['heralds'][10] == 0 and payload['heralds'][11] == 0
    assert payload['heralds'][12] == 0 and payload['heralds'][13] == 0

    # ... but a post-selection function was added
    assert 'postselect' in payload
    ps = deserialize(payload['postselect'])
    assert ps == PostSelect("[0,1]==1 & [2,3]==1 & [4,5]==1 & [6,7]==1")
