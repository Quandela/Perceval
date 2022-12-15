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

from perceval.algorithm.sampler import Sampler
import perceval as pcvl
from perceval.components import BS

import numpy as np


def test_sampler():
    theta_r13 = BS.r_to_theta(1 / 3)
    cnot = pcvl.Circuit(6, name="Ralph CNOT")
    cnot.add((0, 1), BS.H(theta=theta_r13, phi_bl=np.pi, phi_tr=np.pi / 2, phi_tl=-np.pi / 2))
    cnot.add((3, 4), BS.H())
    cnot.add((2, 3), BS.H(theta=theta_r13, phi_bl=np.pi, phi_tr=np.pi / 2, phi_tl=-np.pi / 2))
    cnot.add((4, 5), BS.H(theta=theta_r13))
    cnot.add((3, 4), BS.H())
    imperfect_source = pcvl.Source(brightness=0.9)

    for backend_name in ['CliffordClifford2017', 'Naive', 'SLOS', 'MPS']:
        p = pcvl.Processor(backend_name, cnot, imperfect_source)
        p.mode_post_selection(1)
        p.with_input(pcvl.BasicState([1, 0, 1, 0, 1, 0]))
        sampler = Sampler(p)
        probs = sampler.probs()
        assert probs['results'][pcvl.BasicState('|0,1,2,0,0,0>')] > 0.1
        assert probs['results'][pcvl.BasicState('|0,1,0,0,2,0>')] > 0.1
        samples = sampler.samples(50)
        assert len(samples['results']) == 50
        sample_count = sampler.sample_count(1000)
        assert 950 < sum(list(sample_count['results'].values())) < 1050
