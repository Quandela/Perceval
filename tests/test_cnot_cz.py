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


def add_cnot(processor: pcvl.Processor, ctrl: int, target: int):
    temp_cnot = catalog['heralded cnot'].build_processor()
    processor.add([ctrl * 2, 1 + ctrl * 2, target * 2, 1 + target * 2], temp_cnot)


def add_cnotp(processor: pcvl.Processor, ctrl: int, target: int):
    temp_cnot = catalog['postprocessed cnot'].build_processor()
    processor.add([ctrl * 2, 1 + ctrl * 2, target * 2, 1 + target * 2], temp_cnot)


def add_cz(processor: pcvl.Processor, ctrl: int, target: int):
    temp_cnot = catalog['heralded cz'].build_processor()
    processor.add([ctrl * 2, 1 + ctrl * 2, target * 2, 1 + target * 2], temp_cnot)


def add_x(processor: pcvl.Processor, wire: int):
    circ_x = comp.PERM([1, 0])
    processor.add(2 * wire, circ_x)


def add_h(processor: pcvl.Processor, wire: int):
    processor.add(2 * wire, comp.BS.H())


def measure_processor(processor: pcvl.Processor, expected: pcvl.BasicState):
    init_state = [1, 0] * int(processor.m / 2)
    basic_state = pcvl.BasicState(init_state)

    processor.with_input(basic_state)
    sampler = pcvl.algorithm.Sampler(processor)
    nb_samples = 10
    samples = sampler.samples(nb_samples)

    samples_counter = {}
    for state in samples['results']:
        if state not in samples_counter:
            samples_counter[state] = samples['results'].count(state)

    for (sample, coef) in samples_counter.items():
        # ensure we only have one possibility because it is a deterministic choice.
        assert nb_samples == coef

        assert expected == sample


def test_basic():
    print("Testing Basic 000 measurement...")
    processor = pcvl.Processor("SLOS", 2 * 3)

    expected = pcvl.BasicState([1, 0, 1, 0, 1, 0])
    measure_processor(processor, expected)


def test_cnot():
    print("Testing cnot ctrl=0...")
    processor = pcvl.Processor("SLOS", 2 * 2)

    add_cnot(processor, 0, 1)

    expected = pcvl.BasicState([1, 0, 1, 0])
    measure_processor(processor, expected)


def test_cnot_rev():
    print("Testing cnot ctrl=0 rev...")
    processor = pcvl.Processor("SLOS", 2 * 2)

    add_cnot(processor, 1, 0)

    expected = pcvl.BasicState([1, 0, 1, 0])
    measure_processor(processor, expected)


def test_cnot_rev2():
    print("Testing cnot ctrl=1 rev...")
    processor = pcvl.Processor("SLOS", 2 * 2)
    add_x(processor, 0)
    add_cnot(processor, 1, 0)

    expected = pcvl.BasicState([0, 1, 1, 0])
    measure_processor(processor, expected)


def test_cnot_rev2_czbased():
    print("Testing cnot (based on CZ) ctrl=1 rev...")
    processor = pcvl.Processor("SLOS", 2 * 2)
    add_x(processor, 0)

    add_h(processor, 0)
    add_cz(processor, 1, 0)
    add_h(processor, 0)

    expected = pcvl.BasicState([0, 1, 1, 0])
    measure_processor(processor, expected)


def test_had_and_cz():
    print("Testing Had and native CZ...")
    processor = pcvl.Processor("SLOS", 2 * 3)

    add_x(processor, 0)
    add_h(processor, 1)
    add_h(processor, 2)

    # Flip + to - state
    add_cz(processor, 0, 1)
    add_cz(processor, 0, 2)

    # now convert - to 1
    add_h(processor, 1)
    add_h(processor, 2)

    expected = pcvl.BasicState([0, 1, 0, 1, 0, 1])
    measure_processor(processor, expected)


def test_had_and_cz_cnot_based():
    print("Testing Had and CZ (based on cnot)...")
    processor = pcvl.Processor("SLOS", 2 * 3)

    add_x(processor, 0)
    add_h(processor, 1)
    add_h(processor, 2)

    # Flip + to - state
    add_h(processor, 1)
    add_cnot(processor, 0, 1)
    add_h(processor, 1)

    add_h(processor, 2)
    add_cnot(processor, 0, 2)
    add_h(processor, 2)

    # now convert - to 1
    add_h(processor, 1)
    add_h(processor, 2)

    expected = pcvl.BasicState([0, 1, 0, 1, 0, 1])
    measure_processor(processor, expected)


def test_had_and_cz_rev():
    print("Testing Had and CZ reverse order...")
    processor = pcvl.Processor("SLOS", 2 * 3)

    add_x(processor, 0)
    add_h(processor, 1)
    add_h(processor, 2)

    # Flip + to - state
    add_cz(processor, 1, 0)
    add_cz(processor, 2, 0)

    # now convert - to 1
    add_h(processor, 1)
    add_h(processor, 2)

    expected = pcvl.BasicState([0, 1, 0, 1, 0, 1])
    measure_processor(processor, expected)


def test_had_and_cz_rev_cnotbased():
    print("Testing Had and CZ (based on cnot) reverse order...")
    processor = pcvl.Processor("SLOS", 2 * 3)

    add_x(processor, 0)
    add_h(processor, 1)
    add_h(processor, 2)

    # Flip + to - state using CZ(1,0)
    add_h(processor, 0)
    add_cnot(processor, 1, 0)
    add_h(processor, 0)

    # Flip + to - state using CZ(2,0)
    add_h(processor, 0)
    add_cnot(processor, 2, 0)
    add_h(processor, 0)
    # now convert - to 1
    add_h(processor, 1)
    add_h(processor, 2)

    expected = pcvl.BasicState([0, 1, 0, 1, 0, 1])
    measure_processor(processor, expected)


def test_had_and_cz_rev_cnotbased_post():
    print("Testing Had and CZ (based on cnot post) reverse order...")
    processor = pcvl.Processor("SLOS", 2 * 3)

    add_x(processor, 0)
    add_h(processor, 1)
    add_h(processor, 2)

    # Flip + to - state using CZ(1,0)
    add_h(processor, 0)
    add_cnotp(processor, 1, 0)
    processor.clear_postselection()
    add_h(processor, 0)

    # Flip + to - state using CZ(2,0)
    add_h(processor, 0)
    add_cnotp(processor, 2, 0)
    processor.clear_postselection()
    add_h(processor, 0)

    # now convert - to 1
    add_h(processor, 1)
    add_h(processor, 2)
    processor.set_postselection(pcvl.PostSelect("[0,1]==1 & [2,3]==1 & [4,5]==1"))

    expected = pcvl.BasicState([0, 1, 0, 1, 0, 1])
    measure_processor(processor, expected)
