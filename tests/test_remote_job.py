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
"""module test remote job"""

import pytest
from unittest.mock import patch
from time import sleep

import perceval as pcvl
from perceval.algorithm import Sampler
from perceval.runtime import RemoteJob, RunningStatus
from perceval.runtime.rpc_handler import RPCHandler
from perceval.utils.dist_metrics import tvd_dist
from perceval.utils.conversion import sample_count_to_probs

from _test_utils import LogChecker
from _mock_rpc_handler import RPCHandlerResponsesBuilder, get_rpc_handler_for_tests

SIMPLE_PAYLOAD = {"command": "probs", "circuit": ":PCVL:zip:eJyzCnAO87FydM4sSi7NLLFydfTM9K9wdI7MSg52DsyO9AkNCtWu9DANqMj3cg50hAPP9GwvBM+xEKgWwXPxRFNrEegYlu/jDNTj7mzoGhZQnGEWYkF1ewCY7jxM",
                  "input_state": ":PCVL:BasicState:|1,1>", "parameters": {"min_detected_photons": 2}, "max_shots": 10000, "job_context": None}


@patch.object(pcvl.utils.logging.ExqaliburLogger, "warn")
def test_remote_job(mock_warn):
    rpc_handler = RPCHandler("sim:test", "https://test", "test_token")

    # RemoteJob init
    job_name = 'my_job'
    rj = RemoteJob({"payload": SIMPLE_PAYLOAD}, rpc_handler, job_name)
    assert rj.name == job_name

    for job_name in [None, 42]:
        with pytest.raises(TypeError):
            rj = RemoteJob({"payload": SIMPLE_PAYLOAD}, rpc_handler, job_name)

    rpc_handler_responses_builder = RPCHandlerResponsesBuilder(rpc_handler)

    # SUCCESS
    rpc_handler_responses_builder.set_default_job_status(RunningStatus.SUCCESS)
    rj = RemoteJob({"payload": SIMPLE_PAYLOAD}, rpc_handler, 'my_job')
    rj.execute_async()
    assert rj.is_complete
    with pytest.raises(RuntimeError):
        rj.rerun()
    rj.get_results()  # no throw

    # UNKNOWN
    rpc_handler_responses_builder.set_default_job_status(RunningStatus.UNKNOWN)
    rj = RemoteJob({"payload": SIMPLE_PAYLOAD}, rpc_handler, 'my_job')
    rj.execute_async()
    assert rj.status.unknown

    with LogChecker(mock_warn):
        with pytest.raises(RuntimeError):
            rj.get_results()

    # CANCELED
    rpc_handler_responses_builder.set_default_job_status(RunningStatus.CANCELED)
    rj = RemoteJob({"payload": SIMPLE_PAYLOAD}, rpc_handler, 'my_job')
    rj.execute_async()
    assert rj.status.canceled
    assert rj.status.stop_message == 'Cancel requested from web interface'
    new_rj = rj.rerun()
    assert new_rj.id != rj.id


@patch.object(pcvl.utils.logging.ExqaliburLogger, "warn")
@pytest.mark.parametrize('catalog_item', ["klm cnot", "heralded cnot", "postprocessed cnot", "heralded cz"])
def test_mock_remote_with_gates(mock_warn, catalog_item):
    """test mock remote with gates"""
    noise = pcvl.NoiseModel(
        g2=0.003, transmittance=0.06, phase_imprecision=0, indistinguishability=0.92)
    p = pcvl.catalog[catalog_item].build_processor()
    p.noise = noise
    rp = pcvl.RemoteProcessor.from_local_processor(
        p, rpc_handler=get_rpc_handler_for_tests()
    )

    assert p.heralds == rp.heralds
    assert p.post_select_fn == rp.post_select_fn
    assert p._noise == rp._noise
    assert noise == rp._noise

    for i, input_state in enumerate([pcvl.BasicState(state) for state in [[0, 1, 0, 1], [0, 1, 1, 0], [1, 0, 0, 1], [1, 0, 1, 0]]]):
        if i == 0:
            with LogChecker(mock_warn) as warn_log_checker:
                p.with_input(input_state)
            with warn_log_checker:
                rp.with_input(input_state)
        else:
            p.with_input(input_state)
            rp.with_input(input_state)

        assert p._input_state == rp._input_state


@pytest.mark.skip(reason="need a token and a worker available")
@pytest.mark.parametrize('catalog_item', ["klm cnot", "heralded cnot", "postprocessed cnot", "heralded cz"])
def test_remote_with_gates_probs(catalog_item):
    noise = pcvl.NoiseModel(
        g2=0.003, transmittance=0.06, phase_imprecision=0, indistinguishability=0.92)
    p = pcvl.catalog[catalog_item].build_processor()
    p.min_detected_photons_filter(2 + list(p.heralds.values()).count(1))
    p.noise = noise

    rp = pcvl.RemoteProcessor.from_local_processor(p, 'sim:altair', url='https://api.cloud.quandela.com')

    # platform parameters
    p.thresholded_output(True)
    max_shots_per_call = 1E7

    assert p.heralds == rp.heralds
    assert p.post_select_fn == rp.post_select_fn
    assert p._noise == rp._noise
    assert noise == rp._noise

    for input_state in [pcvl.BasicState(state) for state in [[0, 1, 0, 1], [0, 1, 1, 0], [1, 0, 0, 1], [1, 0, 1, 0]]]:
        rp.with_input(input_state)
        rs = Sampler(rp, max_shots_per_call=max_shots_per_call)
        job = rs.probs.execute_async()

        p.with_input(input_state)
        s = Sampler(p, max_shots_per_call=max_shots_per_call)
        probs = s.probs()

        delay = 0
        while True:
            if job.is_complete:
                break
            assert not job.is_failed
            if delay == 20:
                assert False, "timeout for job"
            delay += 1
            sleep(1)

        tvd = tvd_dist(probs['results'], job.get_results()['results'])
        assert tvd == pytest.approx(0.0, abs=0.2)  # total variation between two distributions is less than 0.2


@pytest.mark.skip(reason="need a token and a worker available")
@pytest.mark.parametrize('catalog_item', ["klm cnot", "heralded cnot", "postprocessed cnot", "heralded cz"])
def test_remote_with_gates_samples(catalog_item):
    noise = pcvl.NoiseModel(
        g2=0.003, transmittance=0.06, phase_imprecision=0, indistinguishability=0.92)
    p = pcvl.catalog[catalog_item].build_processor()
    p.min_detected_photons_filter(2 + list(p.heralds.values()).count(1))
    p.noise = noise
    rp = pcvl.RemoteProcessor.from_local_processor(
        p, "sim:altair", url='https://api.cloud.quandela.com')

    # platform parameters
    p.thresholded_output(True)
    max_shots_per_call = 1E7
    nsamples = 1000

    assert p.heralds == rp.heralds
    assert p.post_select_fn == rp.post_select_fn
    assert p._noise == rp._noise
    assert noise == rp._noise

    for input_state in [pcvl.BasicState(state) for state in [[0, 1, 0, 1], [0, 1, 1, 0], [1, 0, 0, 1], [1, 0, 1, 0]]]:
        rp.with_input(input_state)
        rs = Sampler(rp, max_shots_per_call=max_shots_per_call)
        job = rs.sample_count.execute_async(nsamples)

        p.with_input(input_state)
        s = Sampler(p, max_shots_per_call=max_shots_per_call)
        samples = s.sample_count(nsamples)

        delay = 0
        while True:
            if job.is_complete:
                break
            assert not job.is_failed
            if delay == 20:
                assert False, "timeout for job"
            delay += 1
            sleep(1)

        local_sim_bsd = sample_count_to_probs(samples['results'])
        remote_sim_bsd = sample_count_to_probs(job.get_results()['results'])
        tvd = tvd_dist(local_sim_bsd, remote_sim_bsd)

        assert tvd == pytest.approx(0.0, abs=0.2)  # total variation between two distributions is less than 0.2
