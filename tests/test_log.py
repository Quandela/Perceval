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
from unittest.mock import patch

import perceval as pcvl
from perceval.utils.logging import ExqaliburLogger, PythonLogger, level, channel

from _mock_persistent_data import LoggerConfigForTest
from _mock_rpc_handler import get_rpc_handler

DEFAULT_CONFIG = {'use_python_logger': False, 'enable_file': False,
                  'channels': {'general': {'level': 'off'}, 'resources': {'level': 'off'}, 'user': {'level': 'warn'}}}


@patch.object(ExqaliburLogger, "apply_config")
def test_logger_config(mock_apply_config):
    logger_config = LoggerConfigForTest()
    logger_config.reset()
    logger_config.save()
    assert dict(logger_config) == DEFAULT_CONFIG

    config = logger_config._persistent_data.load_config()
    assert config["logging"] == DEFAULT_CONFIG
    pcvl.get_logger().apply_config(logger_config)
    assert dict(mock_apply_config.call_args[0][0]) == DEFAULT_CONFIG

    logger_config.enable_file()
    logger_config.set_level(level.warn, channel.general)
    logger_config.set_level(level.warn, channel.resources)
    logger_config.set_level(level.warn, channel.user)
    logger_config.save()

    config = logger_config._persistent_data.load_config()
    new_dict_config = {'use_python_logger': False, 'enable_file': True,
                       'channels': {'general': {'level': 'warn'}, 'resources': {'level': 'warn'}, 'user': {'level': 'warn'}}}
    assert config["logging"] == new_dict_config
    pcvl.get_logger().apply_config(logger_config)
    assert dict(mock_apply_config.call_args[0][0]) == new_dict_config

    logger_config.reset()
    logger_config.save()

    config = logger_config._persistent_data.load_config()
    assert config["logging"] == DEFAULT_CONFIG


def test_change_logger():
    pcvl.use_perceval_logger()
    assert isinstance(pcvl.get_logger(), ExqaliburLogger)
    pcvl.use_python_logger()
    assert isinstance(pcvl.get_logger(), PythonLogger)
    pcvl.use_perceval_logger()
    assert isinstance(pcvl.get_logger(), ExqaliburLogger)
    # This test NEEDS to finish on a use_perceval_logger for all other mocked tests to work


def _get_last_dict_logged(mock_info_args):
    return json.loads(mock_info_args)


SOURCE = 'source'
NOISE = 'noise'
LAYER = 'layer'
N = 'n'
M = 'm'
BACKEND = 'backend'
METHOD = 'method'


@patch.object(ExqaliburLogger, "info")
def test_log_resources(mock_info, requests_mock):
    pcvl.utils.logging._logger.set_level(level.info, channel.resources)

    # prepare test parameters
    input_state = pcvl.BasicState("|1,1,0,0>")
    circuit = pcvl.Circuit(4)
    noise_model = pcvl.NoiseModel(brightness=0.2, indistinguishability=0.75, g2=0.05)
    source = pcvl.Source.from_noise_model(noise_model)
    max_samples = 500
    min_detected_photons_filter = 2

    proc_slos = pcvl.Processor('SLOS', circuit, source=source)
    proc_slos.min_detected_photons_filter(min_detected_photons_filter)
    proc_slos.with_input(input_state)
    proc_slos.probs()

    # Processor
    my_dict = _get_last_dict_logged(mock_info.mock_calls[-1].args[0])
    assert my_dict[SOURCE] == source.__dict__()
    assert my_dict[LAYER] == 'Processor'
    assert my_dict[BACKEND] == 'SLOS'
    assert my_dict[N] == input_state.n
    assert my_dict[M] == circuit.m
    assert my_dict[METHOD] == 'probs'

    proc_slos.noise = noise_model
    proc_slos.probs()
    my_dict = _get_last_dict_logged(mock_info.mock_calls[-1].args[0])
    assert SOURCE not in my_dict
    assert my_dict[NOISE] == noise_model.__dict__()
    assert my_dict[LAYER] == 'Processor'
    assert my_dict[BACKEND] == 'SLOS'
    assert my_dict[N] == input_state.n
    assert my_dict[M] == circuit.m
    assert my_dict[METHOD] == 'probs'

    remote_processor = pcvl.RemoteProcessor.from_local_processor(
        proc_slos, rpc_handler=get_rpc_handler(requests_mock)
    )
    remote_processor.with_input(input_state)
    remote_processor.prepare_job_payload('probs')
    my_dict = _get_last_dict_logged(mock_info.mock_calls[-1].args[0])
    assert SOURCE not in my_dict
    assert my_dict['platform'] == 'mocked:platform'
    assert my_dict[NOISE] == noise_model.__dict__()
    assert my_dict[LAYER] == 'RemoteProcessor'
    assert my_dict[N] == input_state.n
    assert my_dict[M] == circuit.m
    assert my_dict['command'] == 'probs'

    proc_clicli = pcvl.Processor('CliffordClifford2017', pcvl.Circuit(4), noise=noise_model)
    proc_clicli.min_detected_photons_filter(min_detected_photons_filter)
    proc_clicli.with_input(input_state)
    proc_clicli.samples(max_samples)


@patch.object(ExqaliburLogger, "info")
def test_log_resources_simulator(mock_info, requests_mock):
    pcvl.get_logger().set_level(level.info, channel.resources)

    # prepare test parameters
    input_state = pcvl.BasicState("|1,1,0,0>")
    circuit = pcvl.Circuit(4)
    noise_model = pcvl.NoiseModel(brightness=0.2, indistinguishability=0.75, g2=0.05)
    source = pcvl.Source.from_noise_model(noise_model)
    input_state_svd = source.generate_distribution(input_state)
    min_detected_photons_filter = 2

    # Simulator
    sim = pcvl.Simulator(pcvl.SLOSBackend())
    sim.set_selection(min_detected_photons_filter=min_detected_photons_filter)
    sim.set_circuit(circuit)
    sim.evolve(input_state)
    my_dict = _get_last_dict_logged(mock_info.mock_calls[-1].args[0])
    assert SOURCE not in my_dict
    assert NOISE not in my_dict
    assert my_dict[LAYER] == 'Simulator'
    assert my_dict[BACKEND] == 'SLOS'
    assert my_dict[N] == input_state.n
    assert my_dict[M] == circuit.m
    assert my_dict[METHOD] == 'evolve'

    sim.probs_svd(input_state_svd)
    my_dict = _get_last_dict_logged(mock_info.mock_calls[-1].args[0])
    assert SOURCE not in my_dict
    assert NOISE not in my_dict
    assert my_dict[LAYER] == 'Simulator'
    assert my_dict[BACKEND] == 'SLOS'
    assert my_dict[N] == input_state_svd.n_max
    assert my_dict[M] == circuit.m
    assert my_dict[METHOD] == 'probs_svd'


@patch.object(ExqaliburLogger, "info")
def test_log_resources_noisy_sampling_simulator(mock_info, requests_mock):
    pcvl.get_logger().set_level(level.info, channel.resources)

    # prepare test parameters
    input_state = pcvl.BasicState("|1,1,0,0>")
    circuit = pcvl.Circuit(4)
    noise_model = pcvl.NoiseModel(brightness=0.2, indistinguishability=0.75, g2=0.05)
    source = pcvl.Source.from_noise_model(noise_model)
    input_state_svd = source.generate_distribution(input_state)
    max_samples = 500
    min_detected_photons_filter = 2

    # Noisy Simulator Simulator
    sim = pcvl.simulators.NoisySamplingSimulator(pcvl.Clifford2017Backend())
    sim.set_selection(min_detected_photons_filter=min_detected_photons_filter)
    sim.set_circuit(circuit)
    sim.samples(input_state_svd, max_samples)
    my_dict = _get_last_dict_logged(mock_info.mock_calls[-1].args[0])
    assert SOURCE not in my_dict
    assert NOISE not in my_dict
    assert my_dict[LAYER] == 'NoisySamplingSimulator'
    assert my_dict[BACKEND] == 'CliffordClifford2017'
    assert my_dict[N] == input_state_svd.n_max
    assert my_dict[M] == circuit.m
    assert my_dict[METHOD] == 'samples'
    assert my_dict['max_samples'] == max_samples
