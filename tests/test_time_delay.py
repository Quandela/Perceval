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
from perceval import Processor, Source, BS, TD, BasicState, BSDistribution

p = Processor("SLOS", 2, source=Source(1))

p.add(0, BS())
p.add(0, TD(1))
p.add(0, BS())

def test_without_herald():
    p.with_input(BasicState([1, 0]))
    p.mode_post_selection(0)

    expected = BSDistribution()
    expected[BasicState([0, 0])] = 0.25
    expected[BasicState([1, 0])] = 0.25
    expected[BasicState([0, 1])] = 0.25
    expected[BasicState([2, 0])] = 0.125
    expected[BasicState([0, 2])] = 0.125

    assert pytest.approx(p.probs()["results"]) == expected, "Basic time delay test not successful"


def test_with_selection():
    p.with_input(BasicState([1, 0]))
    p.mode_post_selection(1)

    expected = BSDistribution()
    expected[BasicState([1, 0])] = 1/3
    expected[BasicState([0, 1])] = 1/3
    expected[BasicState([2, 0])] = 1/6
    expected[BasicState([0, 2])] = 1/6

    expected_p_perf = 3/4

    res = p.probs()

    assert pytest.approx(res["results"]) == expected, "Time delay with selection not successful"
    assert pytest.approx(res["physical_perf"]) == expected_p_perf, "Wrong logical performance with time delays"


def test_with_heralds():
    p.add_herald(1, 0)
    p.with_input(BasicState([1]))
    p.mode_post_selection(0)

    expected = BSDistribution()
    expected[BasicState([0])] = 0.4
    expected[BasicState([1])] = 0.4
    expected[BasicState([2])] = 0.2

    expected_l_perf = 5/8

    res = p.probs()

    assert pytest.approx(res["results"]) == expected, "Time delay with heralds not successful"
    assert pytest.approx(res["logical_perf"]) == expected_l_perf, "Wrong logical performance with time delays"
