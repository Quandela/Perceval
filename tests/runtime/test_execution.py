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

from perceval import LocalComputer, Execution, Computation, Experiment, NoiseModel, RunningStatus, SimulatedComputer
import perceval as pcvl

# This test file is heavily inspired and copied from the tests on the old Job class

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

    def quadratic_count_down(self, _: Experiment, n: int, period: float = 0., must_fail: bool = False, progress_callback=None):
        # We follow the interface for LocalComputer, but we could have avoided the use of self and Experiment by not following it
        l = []
        for i in range(n):
            time.sleep(period)
            if progress_callback:
                progress_callback(i / n, "counting %d" % i)
            l.append(i ** 2)
        assert not must_fail, "Expected failure"  # Dummy failure condition
        return {"results": l}


@pytest.fixture
def execution():
    computer = ComputerForTest()
    return Execution(Computation(computer.get_command("quadratic_count_down"), Experiment()),
                     computer)

def test_run_sync_0(execution):
    assert not execution.was_sent
    assert execution.is_waiting

    assert execution(5) == {"results": [0, 1, 4, 9, 16]}
    assert execution.is_complete
    assert execution.was_sent


@pytest.mark.long_test
def test_run_sync_1(execution):
    all_progress = []
    def progress_callback(progress, message):
        if "counting" in message:  # Ignore regular computer messages
            all_progress.append((progress, message))
    execution.set_progress_callback(progress_callback)

    n = 5
    time_period = 0.01
    assert execution.execute_sync(n, time_period) == {"results": [0, 1, 4, 9, 16]}
    assert len(all_progress) == n
    assert execution.is_complete
    assert execution.status.success
    assert execution.get_results() == {"results": [0, 1, 4, 9, 16]}
    assert len(all_progress) == n  # No more calls
    # Each iteration sleeps for
    assert execution.status.running_time > time_period * n
    assert execution.status.status == RunningStatus.SUCCESS


@pytest.mark.long_test
def test_run_async(execution):
    n = 5
    new_period = 0.2
    assert execution.execute_async(n, new_period) is execution
    assert not execution.is_complete
    counter = 0
    while not execution.is_complete:
        if counter >= 1:
            assert execution.is_running
        counter += 1
        time.sleep(0.3)
    assert counter > 1
    assert execution.status.success
    assert execution.status.stop_message is None
    assert execution.get_results() == {"results": [0, 1, 4, 9, 16]}
    assert execution.status.progress == 1
    # should be at least 1s
    assert execution.status.running_time > new_period * n
    assert execution.status.status == RunningStatus.SUCCESS


@pytest.mark.filterwarnings(f"ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_run_async_fail(execution):
    assert execution.execute_async(5, 0.01, must_fail = True) is execution

    while not execution.is_complete:
        time.sleep(0.1)
    assert execution.status.progress == pytest.approx(0.8)
    assert execution.status.status == RunningStatus.ERROR
    assert "AssertionError" in execution.status.stop_message
    assert execution.status.running_time < 0.5

    with pytest.raises(RuntimeError):
        assert execution.get_results() == {}

    assert "AssertionError" in execution.get_results(allow_partial_results=True)["results"]


def test_run_async_cancel(execution):
    assert execution.execute_async(5, 0.3) is execution
    execution.cancel()
    while execution.is_running:
        time.sleep(0.1)
    assert execution.status.status == RunningStatus.CANCELED


def test_get_res_run_async():
    u = pcvl.Unitary.random(6)
    bs = pcvl.BasicState("|1,0,1,0,1,0>")
    e = Experiment(u)
    e.with_input(bs)
    computer = SimulatedComputer("SLOS")
    computation = Computation(computer.get_command("sample_count"), e)
    execution = Execution(computation, computer)
    execution.execute_async(10000)
    while not execution.is_complete:
        time.sleep(0.01)

    res_1st_call = execution.get_results()
    res_2nd_call = execution.get_results()

    assert isinstance(res_1st_call["results"], pcvl.BSCount)
    assert isinstance(res_2nd_call["results"], pcvl.BSCount)

    assert res_1st_call["results"] == res_2nd_call["results"]
    assert res_1st_call["global_perf"] == res_2nd_call["global_perf"]
