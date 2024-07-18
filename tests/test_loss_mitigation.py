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
import numpy as np
from perceval.error_mitigation import photon_recycling
from perceval.utils import BasicState, BSDistribution
from perceval.components import catalog, Unitary
from perceval.utils import Matrix, NoiseModel
from perceval.algorithm import Sampler
from perceval import Processor
from perceval.error_mitigation._loss_mitigation_utils import _gen_lossy_dists
from perceval.error_mitigation.loss_mitigation import _generate_one_photon_per_mode_mapping



def _sampler_setup_cnot(output_type: str):
    # Processor config
    processor = Processor("SLOS", noise=NoiseModel(transmittance=0.3))
    processor.min_detected_photons_filter(0)
    processor.thresholded_output(True)

    # Circuit
    circ = catalog['heralded cnot'].build_circuit()
    processor.set_circuit(circ)

    # Input state
    input_state = BasicState([0, 1, 0, 1, 1, 1])
    processor.with_input(input_state)

    # Sampler
    sampler = Sampler(processor)
    if output_type == 'samples':
        return sampler.sample_count(20000)['results']
    elif output_type == 'probs':
        return sampler.probs()['results']

@pytest.mark.parametrize('result_type', ['probs', 'samples'])
def test_photon_loss_mitigation(result_type):

    lossy_sampling = _sampler_setup_cnot(output_type=result_type)
    ideal_photon_count = 4
    mitigated_dist = photon_recycling(lossy_sampling, ideal_photon_count)

    for keys, value in mitigated_dist.items():
        assert sum(keys) == ideal_photon_count
        assert value > 1e-6

def test_input_validation_loss_mitigation():
    bs1 = BasicState([1, 1, 1, 0, 0, 0, 0])
    bs2 = BasicState([1, 0, 0, 1, 1, 0, 0])
    bs3 = BasicState([1, 0, 0, 0, 0, 0, 0])
    bs4 = BasicState([0, 0, 0, 1, 1, 0, 0])

    with pytest.raises(TypeError):
        photon_recycling(bs1, 3)

    # perfect bsd
    perfect_bsd = BSDistribution()
    perfect_bsd[bs1] = 0.9
    perfect_bsd[bs2] = 0.1

    # missing n-1 state
    lossy_bsd1 = BSDistribution()
    lossy_bsd1[bs1] = 0.3333
    lossy_bsd1[bs2] = 0.3333
    lossy_bsd1[bs3] = 0.3333

    # missing n-2 state
    lossy_bsd2 = BSDistribution()
    lossy_bsd2[bs1] = 0.3333
    lossy_bsd2[bs2] = 0.3333
    lossy_bsd2[bs4] = 0.3333

    with pytest.raises(ValueError):
        photon_recycling(perfect_bsd, 3)

    with pytest.raises(ValueError):
        photon_recycling(lossy_bsd1, 3)

    with pytest.raises(ValueError):
        photon_recycling(lossy_bsd2, 3)

    with pytest.raises(ValueError):
        photon_recycling(lossy_bsd1, 6)  # incorrect count

    with pytest.raises(ValueError):
        photon_recycling(lossy_bsd2, 6)

def _compute_random_circ_probs(source_emission, num_photons):

    random_loc = Unitary(Matrix.random_unitary(20))
    # Processor config
    processor = Processor("SLOS", random_loc, noise=NoiseModel(transmittance=source_emission))
    processor.min_detected_photons_filter(0)
    processor.thresholded_output(True)

    # Input state
    input_state = BasicState([1] * num_photons + [0] * (random_loc.m - num_photons))

    processor.with_input(input_state)

    # Sampler
    sampler = Sampler(processor)
    return sampler.probs()['results']

def total_variation_distance(p1, p2):
    # calculate Total Variation distance between 2 probability distributions
    tvd = 0.5 * np.sum(np.abs(np.array(p1) - np.array(p2)))
    return tvd

def test_mitigation_over_postselect_tvd():

    ideal_photon_count = 4
    # lossless distribution
    ideal_dist = _compute_random_circ_probs(source_emission=1, num_photons=ideal_photon_count)
    # lossy distribution
    lossy_dist = _compute_random_circ_probs(source_emission=0.3, num_photons=ideal_photon_count)

    # compute the migitated distribution from
    mitigated_dist = photon_recycling(lossy_dist, ideal_photon_count)

    # computing the postselected distribution for this random computation.
    # Using the internal methods of loss_mitigation codes for this
    # as the PostSelect function to set on Processor is not known

    num_modes = next(iter(lossy_dist)).m  # number of modes
    pattern_map = _generate_one_photon_per_mode_mapping(num_modes, ideal_photon_count)
    noisy_distributions = _gen_lossy_dists(lossy_dist, ideal_photon_count, pattern_map)
    post_dist = noisy_distributions[0]

    for key, values in ideal_dist.items():
        if key not in mitigated_dist.keys():
            mitigated_dist[key] = 0.0

    # computing tvd
    tvd_miti = total_variation_distance(list(ideal_dist.values()), list(mitigated_dist.values()))
    tvd_ps = total_variation_distance(list(ideal_dist.values()), list(post_dist))

    assert tvd_miti < tvd_ps  # checks that mitigated is closer to ideal than post-selected distribution
