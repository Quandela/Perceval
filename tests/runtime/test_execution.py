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

import time

import pytest

from perceval import LocalComputer, Execution, Computation, Experiment, NoiseModel

# This test file is heavily inspired by the test on the old Job class

PERIOD = 0.1

class ComputerForTest(LocalComputer):

    @property
    def noise(self):
        return NoiseModel()

    @noise.setter
    def noise(self, noise: NoiseModel):
        pass

    @property
    def performance(self):
        pass

    @property
    def type(self):
        pass

    def __init__(self):
        super().__init__()
        self._register_method(ComputerForTest.quadratic_count_down, use_emt=False)

    def quadratic_count_down(self, _: Experiment, n: int, period: float = PERIOD, must_fail: bool = False, progress_callback=None):
        l = []
        for i in range(n):
            time.sleep(period)
            if progress_callback:
                progress_callback(i / n, "counting %d" % i)
            l.append(i ** 2)
        assert not must_fail  # Dummy failure condition
        return {"results": l}


@pytest.fixture
def execution():
    computer = ComputerForTest()
    return Execution(Computation(computer.get_command("quadratic_count_down"), Experiment()),
                     computer)

def test_run_sync_0(execution):
    assert execution(5, 0.) == {"results": [0, 1, 4, 9, 16]}
