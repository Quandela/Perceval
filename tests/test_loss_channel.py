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
import perceval as pcvl

U = pcvl.Matrix.random_unitary(2)
simulator = pcvl.BackendFactory().get_backend("Stepper")
loss = .3

cd = (pcvl.Circuit(2)
      .add(0, pcvl.Unitary(U))
      .add(0, pcvl.LC(loss))
      .add(1, pcvl.LC(loss)))


def test_minimal():
    sim = simulator(pcvl.LC(loss))
    input_state = pcvl.BasicState([2])
    expected = {
        pcvl.BasicState([0]): loss ** 2,
        pcvl.BasicState([1]): 2 * loss * (1 - loss),
        pcvl.BasicState([2]): (1 - loss) ** 2
    }
    res = {state: prob
           for state, prob in sim.allstateprob_iterator(input_state)
           }

    assert pytest.approx(res) == expected


def test_permutation():
    input_state = pcvl.BasicState([1, 1])

    cg = (pcvl.Circuit(2)
          .add(0, pcvl.LC(loss))
          .add(1, pcvl.LC(loss))
          .add(0, pcvl.Unitary(U)))

    assert pytest.approx({state: prob
                          for state, prob in simulator(cg).allstateprob_iterator(input_state)
                          }) == {state: prob
                                 for state, prob in simulator(cd).allstateprob_iterator(input_state)
                                 }


def test_brightness_equivalence():
    input_state = pcvl.BasicState([1, 1])
    source = pcvl.Source(brightness=1 - loss)
    p = pcvl.Processor("SLOS", pcvl.Unitary(U), source)

    p.with_input(input_state)
    p.mode_post_selection(0)

    sampler = pcvl.algorithm.Sampler(p)
    real_out = sampler.probs()["results"]

    real_out = {
        state[0]: prob for state, prob in real_out.items()
    }

    sim = simulator(cd)

    assert pytest.approx(real_out) == {state: prob
                                       for state, prob in sim.allstateprob_iterator(input_state)
                                       }
