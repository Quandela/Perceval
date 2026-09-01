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
"""module test payload generation"""
import pytest

from perceval import RemoteProcessor, BasicState, catalog, PayloadGenerator, SimulatedComputer, NoiseModel
from perceval.serialization._constants import ZIP_PREFIX

from tests.providers.quandela._mock_rpc_handler import get_rpc_handler_for_tests


COMMAND_NAME = 'my_command'


def _get_remote_processor(m: int = 8):
    return RemoteProcessor(rpc_handler=get_rpc_handler_for_tests(), m=m)


def test_payload_basics():
    """test payload basics infos"""
    rp = _get_remote_processor()
    rp.min_detected_photons_filter(4)
    cloud_data = rp.prepare_job_payload(COMMAND_NAME)
    assert 'platform_name' in cloud_data and cloud_data['platform_name'] == rp._rpc_handler.name
    assert 'pcvl_version' in cloud_data
    assert 'process_id' in cloud_data
    assert 'payload' in cloud_data

    payload = cloud_data['payload']
    assert 'command' in payload and payload['command'] == COMMAND_NAME
    assert 'experiment' in payload and payload['experiment'].startswith(ZIP_PREFIX)  # Circuits are compressed in payloads


def test_payload_parameters():
    """test parameters of payload"""
    n_params = 5
    rp = _get_remote_processor()
    params = {f'param{i}': f'value{i}' for i in range(n_params)}
    rp.set_parameters(params)

    rp.min_detected_photons_filter(0)
    payload = rp.prepare_job_payload(COMMAND_NAME)['payload']
    assert 'parameters' in payload
    for i in range(n_params):
        assert f'param{i}' in payload['parameters']
        assert payload['parameters'][f'param{i}'] == f'value{i}'

def test_payload_cnot():
    """test payload with cnot"""
    rp = _get_remote_processor()
    heralded_cnot = catalog['heralded cnot'].build_experiment()
    pp_cnot = catalog['postprocessed cnot'].build_experiment()
    rp.add(0, heralded_cnot)
    rp.add(0, pp_cnot)
    rp.add(4, pp_cnot)
    assert rp.m == 8
    assert rp.circuit_size == 14  # 8 modes of interest + 6 ancillaries

    input_state = BasicState([1, 0] * 4)
    rp.with_input(input_state)

    with pytest.raises(ValueError):
        rp.prepare_job_payload(COMMAND_NAME)  # Missing min_detected_photons

    rp.min_detected_photons_filter(4)

    cloud_data = rp.prepare_job_payload(COMMAND_NAME)
    assert 'platform_name' in cloud_data and cloud_data['platform_name'] == rp._rpc_handler.name
    assert 'pcvl_version' in cloud_data
    assert 'process_id' in cloud_data
    assert 'payload' in cloud_data

    payload = cloud_data['payload']
    assert 'command' in payload and payload['command'] == COMMAND_NAME
    assert 'experiment' in payload and payload['experiment'].startswith(
        ZIP_PREFIX)  # Circuits are compressed in payloads

def test_payload_generator():
    cloud_data = PayloadGenerator.generate_cloud_data(COMMAND_NAME)

    assert 'pcvl_version' in cloud_data
    assert 'process_id' in cloud_data
    assert 'payload' in cloud_data

    payload = cloud_data['payload']
    assert 'command' in payload and payload['command'] == COMMAND_NAME
    assert 'experiment' in payload and payload['experiment'].startswith(ZIP_PREFIX)  # Circuits are compressed in payloads


def test_payload_applier():
    computer = SimulatedComputer("SLOS")
    default_parameters = {"compute_physical_logical_perf": True}
    computer.parameters = default_parameters

    default_noise =  NoiseModel(0.8)
    computer.noise = default_noise

    in_computation_parameters = {"compute_physical_logical_perf": False}
    in_computation_noise = NoiseModel(0.6)

    payload = PayloadGenerator.from_computation(None, [], in_computation_parameters, in_computation_noise)

    with PayloadGenerator.payload_applier(computer, payload):
        assert computer.noise == in_computation_noise
        assert computer.parameters == in_computation_parameters

    assert computer.noise == default_noise
    assert computer.parameters == default_parameters
