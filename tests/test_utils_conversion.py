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

from perceval.utils.conversion import *
from perceval.utils import BasicState


b0 = BasicState([0, 0, 0, 1])
b1 = BasicState([0, 0, 1, 0])
b2 = BasicState([0, 1, 0, 0])
b3 = BasicState([1, 0, 0, 0])


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


def test_probs_to_sample_count():
    bsd = BSDistribution()
    bsd[b0] = 0.1
    bsd[b1] = 0.25
    bsd[b2] = 0.15
    bsd[b3] = 0.5
    output = probs_to_sample_count(bsd, 1000)
    assert output[b0] < output[b2]
    assert output[b2] < output[b1]
    assert output[b1] < output[b3]
