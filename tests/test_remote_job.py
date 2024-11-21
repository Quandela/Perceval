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
from perceval.utils.dist_metrics import tvd_dist
from perceval.utils.conversion import sample_count_to_probs

from _test_utils import LogChecker
from _mock_rpc_handler import (
    get_rpc_handler,
    REMOTE_JOB_DURATION,
    REMOTE_JOB_RESULTS,
    REMOTE_JOB_CREATION_TIMESTAMP,
    REMOTE_JOB_START_TIMESTAMP,
    REMOTE_JOB_NAME,
)


@patch.object(pcvl.utils.logging.ExqaliburLogger, "warn")
def test_remote_job(mock_warn, requests_mock):
    """test remote job"""
    _FIRST_JOB_NAME = 'job name'
    _SECOND_JOB_NAME = 'another name'
    rj = RemoteJob({}, get_rpc_handler(requests_mock), _FIRST_JOB_NAME)
    assert rj.name == _FIRST_JOB_NAME
    rj.name = _SECOND_JOB_NAME
    assert rj.name == _SECOND_JOB_NAME
    with pytest.raises(TypeError):
        rj.name = None
    with pytest.raises(TypeError):
        rj.name = 28
    job_status = rj.status
    assert rj.is_complete == job_status.completed
    with pytest.raises(RuntimeError):
        rj.rerun()
    assert rj.get_results()['results'] == REMOTE_JOB_RESULTS

    rj.status.status = RunningStatus.UNKNOWN
    with LogChecker(mock_warn):
        assert rj.get_results()['results'] == REMOTE_JOB_RESULTS

    rj.status.status = RunningStatus.CANCELED
    new_rj = rj.rerun()
    assert new_rj.id != rj.id

    _TEST_JOB_ID = "any"
    resumed_rj = RemoteJob.from_id(_TEST_JOB_ID, get_rpc_handler(requests_mock))
    assert resumed_rj.get_results()['results'] == REMOTE_JOB_RESULTS
    assert resumed_rj.id == _TEST_JOB_ID
    assert rj.is_complete == job_status.completed
    assert rj.name == REMOTE_JOB_NAME
    assert rj.status.creation_timestamp == REMOTE_JOB_CREATION_TIMESTAMP
    assert rj.status.start_timestamp == REMOTE_JOB_START_TIMESTAMP
    assert rj.status.duration == REMOTE_JOB_DURATION


@patch.object(pcvl.utils.logging.ExqaliburLogger, "warn")
@pytest.mark.parametrize('catalog_item', ["klm cnot", "heralded cnot", "postprocessed cnot", "heralded cz"])
def test_mock_remote_with_gates(mock_warn, requests_mock, catalog_item):
    """test mock remote with gates"""
    noise = pcvl.NoiseModel(
        g2=0.003, transmittance=0.06, phase_imprecision=0, indistinguishability=0.92)
    p = pcvl.catalog[catalog_item].build_processor()
    p.noise = noise
    rp = pcvl.RemoteProcessor.from_local_processor(
        p, rpc_handler=get_rpc_handler(requests_mock)
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
