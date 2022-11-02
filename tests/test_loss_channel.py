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

import pytest
from perceval import Processor, Unitary, LC, Matrix, BSDistribution, BasicState, Source, SVDistribution
from perceval.algorithm import Sampler


U = Matrix.random_unitary(2)
loss = .3

cd = (Processor("SLOS", 2)
      .add(0, Unitary(U))
      .add(0, LC(loss))
      .add(1, LC(loss)))
input_state = BasicState([1, 1])

cd.with_input(input_state)
cd.mode_post_selection(0)

def test_minimal():
    p = Processor("SLOS", 1).add(0, LC(loss))
    p.with_input(SVDistribution(BasicState([2])))
    p.mode_post_selection(0)
    expected_svd = BSDistribution()
    expected_svd[BasicState([0])] = loss ** 2
    expected_svd[BasicState([1])] = 2 * loss * (1 - loss)
    expected_svd[BasicState([2])] = (1 - loss) ** 2

    res = p.probs()["results"]

    assert pytest.approx(res) == expected_svd


def test_permutation():

    cg = (Processor("SLOS", 2)
          .add(0, LC(loss))
          .add(1, LC(loss))
          .add(0, Unitary(U)))

    cg.with_input(input_state)
    cg.mode_post_selection(0)

    assert pytest.approx(cg.probs()["results"]) == cd.probs()["results"]


def test_brightness_equivalence():
    source = Source(brightness=1 - loss)
    p = Processor("SLOS", Unitary(U), source)

    p.with_input(input_state)
    p.mode_post_selection(0)

    sampler = Sampler(p)
    real_out = sampler.probs()["results"]

    assert pytest.approx(real_out) == cd.probs()["results"]
