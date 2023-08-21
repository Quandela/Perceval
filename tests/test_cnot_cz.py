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

import perceval as pcvl
import perceval.components as comp
from perceval.components import catalog


def cnot(proc, ctrl: int, target: int):
    temp_cnot = catalog['heralded cnot'].as_processor().build()
    proc.add([ctrl * 2, 1 + ctrl * 2, target * 2, 1 + target * 2], temp_cnot)


def cnotp(proc, ctrl: int, target: int):
    temp_cnot = catalog['postprocessed cnot'].as_processor().build()
    proc.add([ctrl * 2, 1 + ctrl * 2, target * 2, 1 + target * 2], temp_cnot)


def cz(proc, ctrl: int, target: int):
    temp_cnot = catalog['heralded cz'].as_processor().build()
    proc.add([ctrl * 2, 1 + ctrl * 2, target * 2, 1 + target * 2], temp_cnot)


def x(proc, wire):
    circ_x = comp.PERM([1, 0])
    proc.add(2 * wire, circ_x)


def h(proc, wire):
    proc.add(2 * wire, comp.BS.H())


def measure_proc(proc, expected, nb_samples=10, post=False):
    init_state = [1, 0] * int(proc.m / 2)
    basic_state = pcvl.BasicState(init_state)

    proc.with_input(basic_state)
    sampler = pcvl.algorithm.Sampler(proc)
    samples = sampler.samples(nb_samples)

    samples_counter = dict()
    for state in samples['results']:
        if samples_counter.get(state) is None:
            samples_counter[state] = 1
        else:
            samples_counter[state] += 1
    if post:
        temp = samples_counter.copy()
        # clear the dict from wrong results
        for (sample, coef) in temp.items():
            l_sample = list(sample)
            if 2 in l_sample:
                samples_counter.pop(sample, None)
            else:
                for i in range(0, len(l_sample), 2):
                    if l_sample[i] == l_sample[i + 1]:
                        samples_counter.pop(sample, None)
    for (sample, coef) in samples_counter.items():
        if not post:
            # ensure we only have one possibility because it is a deterministic choice.
            assert nb_samples == coef

        assert expected == list(sample)


def test_basic():
    print("Testing Basic 000 measurement...")
    proc = pcvl.Processor("SLOS", 2 * 3)

    expected = [1, 0, 1, 0, 1, 0]
    measure_proc(proc, expected)


def test_cnot():
    print("Testing cnot ctrl=0...")
    proc = pcvl.Processor("SLOS", 2 * 2)

    cnot(proc, 0, 1)
    expected = [1, 0, 1, 0]
    measure_proc(proc, expected)


def test_cnot_rev():
    print("Testing cnot ctrl=0 rev...")
    proc = pcvl.Processor("SLOS", 2 * 2)

    cnot(proc, 1, 0)
    expected = [1, 0, 1, 0]
    measure_proc(proc, expected)


def test_cnot_rev2():
    print("Testing cnot ctrl=1 rev...")
    proc = pcvl.Processor("SLOS", 2 * 2)
    x(proc, 0)
    cnot(proc, 1, 0)
    expected = [0, 1, 1, 0]
    measure_proc(proc, expected)


def test_cnot_rev2_czbased():
    print("Testing cnot (based on CZ) ctrl=1 rev...")
    proc = pcvl.Processor("SLOS", 2 * 2)
    x(proc, 0)

    h(proc, 0)
    cz(proc, 1, 0)
    h(proc, 0)

    expected = [0, 1, 1, 0]
    measure_proc(proc, expected)


def test_had_and_cz():
    print("Testing Had and native CZ...")
    proc = pcvl.Processor("SLOS", 2 * 3)

    x(proc, 0)
    h(proc, 1)
    h(proc, 2)

    # Flip + to - state
    cz(proc, 0, 1)
    cz(proc, 0, 2)

    # now convert - to 1
    h(proc, 1)
    h(proc, 2)
    expected = [0, 1, 0, 1, 0, 1]
    measure_proc(proc, expected)


def test_had_and_cz_cnot_based():
    print("Testing Had and CZ (based on cnot)...")
    proc = pcvl.Processor("SLOS", 2 * 3)

    x(proc, 0)
    h(proc, 1)
    h(proc, 2)

    # Flip + to - state
    h(proc, 1)
    cnot(proc, 0, 1)
    h(proc, 1)

    h(proc, 2)
    cnot(proc, 0, 2)
    h(proc, 2)

    # now convert - to 1
    h(proc, 1)
    h(proc, 2)
    expected = [0, 1, 0, 1, 0, 1]
    measure_proc(proc, expected)


def test_had_and_cz_rev():
    print("Testing Had and CZ reverse order...")
    proc = pcvl.Processor("SLOS", 2 * 3)

    x(proc, 0)
    h(proc, 1)
    h(proc, 2)

    # Flip + to - state
    cz(proc, 1, 0)
    cz(proc, 2, 0)

    # now convert - to 1
    h(proc, 1)
    h(proc, 2)
    expected = [0, 1, 0, 1, 0, 1]
    measure_proc(proc, expected)


def test_had_and_cz_rev_cnotbased():
    print("Testing Had and CZ (based on cnot) reverse order...")
    proc = pcvl.Processor("SLOS", 2 * 3)

    x(proc, 0)
    h(proc, 1)
    h(proc, 2)

    # Flip + to - state using CZ(1,0)
    h(proc, 0)
    cnot(proc, 1, 0)
    h(proc, 0)

    # Flip + to - state using CZ(2,0)
    h(proc, 0)
    cnot(proc, 2, 0)
    h(proc, 0)
    # now convert - to 1
    h(proc, 1)
    h(proc, 2)

    expected = [0, 1, 0, 1, 0, 1]
    measure_proc(proc, expected)


def test_had_and_cz_rev_cnotbased_post():
    print("Testing Had and CZ (based on cnot post) reverse order...")
    proc = pcvl.Processor("SLOS", 2 * 3)

    x(proc, 0)
    h(proc, 1)
    h(proc, 2)

    # Flip + to - state using CZ(1,0)
    h(proc, 0)
    cnotp(proc, 1, 0)
    proc.clear_postselection()
    h(proc, 0)

    # Flip + to - state using CZ(2,0)
    h(proc, 0)
    cnotp(proc, 2, 0)
    proc.clear_postselection()
    h(proc, 0)
    # now convert - to 1
    h(proc, 1)
    h(proc, 2)
    proc.set_postselection(pcvl.PostSelect("[0,1]==1 & [2,3]==1 & [4,5]==1"))

    expected = [0, 1, 0, 1, 0, 1]
    measure_proc(proc, expected)
