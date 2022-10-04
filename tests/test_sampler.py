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
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from perceval.algorithm.sampler import *
import perceval as pcvl
from perceval.components import BS

import numpy as np


b0 = BasicState([0, 0, 0, 1])
b1 = BasicState([0, 0, 1, 0])
b2 = BasicState([0, 1, 0, 0])
b3 = BasicState([1, 0, 0, 0])


# Test conversion functions
def test_samples_to_sample_count():
    sample_list = [b0, b1, b2, b3]
    output = samples_to_sample_count(sample_list)
    assert len(output) == 4
    for s in sample_list:
        assert output[s] == 1

    sample_list = [b0, b0, b1, b3, b0, b1, b3, b1, b2, b0, b0, b3, b0]
    output = samples_to_sample_count(sample_list)
    assert len(output) == 4
    assert output[b0] == 6
    assert output[b1] == 3
    assert output[b2] == 1
    assert output[b3] == 3

    assert len(samples_to_sample_count([])) == 0


def test_sample_count_to_probs():
    sample_count = {
        b0: 280,
        b1: 120,
        b2: 400,
        b3: 200
    }
    output = sample_count_to_probs(sample_count)
    assert sum(output.values()) == 1
    assert output[b0] == 0.28
    assert output[b1] == 0.12
    assert output[b2] == 0.4
    assert output[b3] == 0.2

    empty = sample_count_to_probs({})
    assert len(empty) == 0


# Test sampler algorithm
def test_sampler():
    theta_r13 = BS.r_to_theta(1 / 3)
    cnot = pcvl.Circuit(6, name="Ralph CNOT")
    cnot.add((0, 1), BS.H(theta=theta_r13, phi_bl=np.pi, phi_tr=np.pi / 2, phi_tl=-np.pi / 2))
    cnot.add((3, 4), BS.H())
    cnot.add((2, 3), BS.H(theta=theta_r13, phi_bl=np.pi, phi_tr=np.pi / 2, phi_tl=-np.pi / 2))
    cnot.add((4, 5), BS.H(theta=theta_r13))
    cnot.add((3, 4), BS.H())
    imperfect_source = pcvl.Source(brightness=0.9)

    for backend_name in ['CliffordClifford2017', 'Naive', 'SLOS', 'Stepper', 'MPS']:
        p = pcvl.Processor(backend_name, cnot, imperfect_source)
        p.with_input(BasicState([1, 0, 1, 0, 1, 0]))
        sampler = Sampler(p)
        probs = sampler.probs()
        assert probs[pcvl.BasicState('|0,1,2,0,0,0>')] > 0.1
        assert probs[pcvl.BasicState('|0,1,0,0,2,0>')] > 0.1
        samples = sampler.samples(50)
        assert len(samples) == 50
        sample_count = sampler.sample_count(50)
        assert sum(list(sample_count.values())) == 50
