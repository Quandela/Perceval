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

import perceval as pcvl
import time


def quadratic_count_down(n, speed=0.1, progress_callback=None):
    l = []
    for i in range(n):
        time.sleep(speed)
        if progress_callback:
            progress_callback(i/n, "counting %d" % i)
        l.append(i**2)
    assert speed >= 0.1
    return l


def test_run_sync_0():
    assert(pcvl.LocalJob(quadratic_count_down)(5) == [0, 1, 4, 9, 16])


def test_run_sync_1():
    job = pcvl.LocalJob(quadratic_count_down)
    assert job.execute_sync(5) == [0, 1, 4, 9, 16]
    assert job.is_completed()
    assert job.status.success
    assert job.get_results() == [0, 1, 4, 9, 16]
    # should be ~ 0.5 s
    assert job.status.running_time < 1


def test_run_async():
    job = pcvl.LocalJob(quadratic_count_down)
    assert job.execute_async(5, speed=0.3) is job
    assert not job.is_completed()
    counter = 0
    while not job.is_completed():
        counter += 1
        time.sleep(0.5)
    assert counter > 1
    assert job.status.success
    assert job.status.stop_message is None
    assert job.get_results() == [0, 1, 4, 9, 16]
    assert job.status.progress == 1
    # should be at least 1.5s
    assert job.status.running_time > 1

def test_run_async_fail():
    job = pcvl.LocalJob(quadratic_count_down)
    assert job.execute_async(5, speed=0.01) is job
    counter = 0
    while not job.is_completed():
        counter += 1
        time.sleep(1)
    assert not job.status.success
    assert job.status.progress == 0.8
    assert "AssertionError" in job.status.stop_message
    # should be ~0.05 s
    assert job.status.running_time < 0.5
