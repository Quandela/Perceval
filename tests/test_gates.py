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

import pytest

from perceval.components import BS, PERM, catalog
from perceval import Processor, BasicState, algorithm, PostSelect


def sample_processor(processor: Processor, nb_sample: int):
    init_state = [1, 0] * int(processor.m / 2)
    basic_state = BasicState(init_state)

    processor.with_input(basic_state)
    sampler = algorithm.Sampler(processor)

    samples = sampler.samples(nb_sample)

    samples_counter = {}
    for state in samples['results']:
        if state not in samples_counter:
            samples_counter[state] = samples['results'].count(state)
    return samples_counter


def measure_processor(processor: Processor, expected: BasicState):
    nb_sample = 10
    samples_counter = sample_processor(processor, nb_sample)

    for (sample, coef) in samples_counter.items():
        # ensure we only have one possibility because it is a deterministic choice.
        assert nb_sample == coef
        assert expected == sample


def compare_processors(processor_lh: Processor, processor_rh: Processor):
    nb_sample = 10
    samples_counter_lh = sample_processor(processor_lh, nb_sample)
    samples_counter_rh = sample_processor(processor_rh, nb_sample)

    assert samples_counter_lh == samples_counter_rh


def test_cnot_czbased():
    b1010 = BasicState([1, 0, 1, 0])

    processor = Processor("SLOS", 4)
    processor.add((2, 3, 0, 1), catalog['heralded cnot'].build_processor())
    measure_processor(processor, b1010)

    processor = Processor("SLOS", 4)
    processor.add((0, 1, 2, 3), catalog['heralded cnot'].build_processor())  # CNOT
    measure_processor(processor, b1010)

    processor = Processor("SLOS", 4)
    # convert bit to phase
    processor.add((2, 3), BS.H())  # H
    # phase flip
    processor.add((0, 1, 2, 3), catalog['heralded cz'].build_processor())  # CZ
    # convert phase to bit
    processor.add((2, 3), BS.H())  # H

    measure_processor(processor, b1010)


def test_CZ_HCXH():
    processorX = Processor("SLOS", 4)
    processorX.add((0, 1, 2, 3), catalog['heralded cz'].build_processor())

    processorZ = Processor("SLOS", 4)
    processorZ.add((2, 3), BS.H())
    processorZ.add((0, 1, 2, 3), catalog['heralded cnot'].build_processor())
    processorZ.add((2, 3), BS.H())

    compare_processors(processorX, processorZ)

    processorZ = Processor("SLOS", 4)
    processorZ.add((2, 3), BS.H())
    processorZ.add((0, 1, 2, 3), catalog['postprocessed cnot'].build_processor())
    processorZ.clear_postprocess()
    processorZ.add((2, 3), BS.H())

    processorZ.set_postselection(PostSelect("[0,1]==1 & [2,3]==1"))
    compare_processors(processorX, processorZ)


def test_HCZH_CX():
    processorX = Processor("SLOS", 4)
    processorX.add((0, 1, 2, 3), catalog['heralded cnot'].build_processor())

    processorZ = Processor("SLOS", 4)
    processorZ.add((2, 3), BS.H())
    processorZ.add((0, 1, 2, 3), catalog['heralded cz'].build_processor())
    processorZ.add((2, 3), BS.H())

    compare_processors(processorX, processorZ)

    processorX = Processor("SLOS", 4)
    processorX.add((0, 1, 2, 3), catalog['postprocessed cnot'].build_processor())
    processorX.clear_postprocess()

    processorX.set_postselection(PostSelect("[0,1]==1 & [2,3]==1"))
    compare_processors(processorX, processorZ)


def test_had_and_cz():
    b101010 = BasicState([1, 0, 1, 0, 1, 0])

    # Testing CZ
    processor = Processor("SLOS", 6)

    # convert bit to phase
    processor.add((2, 3), BS.H())
    processor.add((4, 5), BS.H())

    # phase flip
    processor.add((0, 1, 2, 3), catalog['heralded cz'].build_processor())
    processor.add((0, 1, 4, 5), catalog['heralded cz'].build_processor())

    # convert phase to bit
    processor.add((2, 3), BS.H())
    processor.add((4, 5), BS.H())

    measure_processor(processor, b101010)

    # -----------
    # Testing CZ (based on cnot)
    processor = Processor("SLOS", 6)

    # convert bit to phase
    processor.add((2, 3), BS.H())
    processor.add((4, 5), BS.H())

    # phase flip

    # convert phase to bit
    processor.add((2, 3), BS.H())
    # bit flip
    processor.add((0, 1, 2, 3), catalog['heralded cnot'].build_processor())
    # convert bit to phase
    processor.add((2, 3), BS.H())

    # convert phase to bit
    processor.add((4, 5), BS.H())
    # bit flip
    processor.add((0, 1, 4, 5), catalog['heralded cnot'].build_processor())
    # convert bit to phase
    processor.add((4, 5), BS.H())

    # convert phase to bit
    processor.add((2, 3), BS.H())
    processor.add((4, 5), BS.H())

    measure_processor(processor, b101010)


def test_had_and_cz_rev():
    b101010 = BasicState([1, 0, 1, 0, 1, 0])
    # Testing Had and CZ reverse order
    processor = Processor("SLOS", 6)

    # convert bit to phase
    processor.add((2, 3), BS.H())
    processor.add((4, 5), BS.H())

    # phase flip
    processor.add((2, 3, 0, 1), catalog['heralded cz'].build_processor())
    processor.add((4, 5, 0, 1), catalog['heralded cz'].build_processor())

    # convert phase to bit
    processor.add((2, 3), BS.H())
    processor.add((4, 5), BS.H())

    measure_processor(processor, b101010)

    # Testing Had and CZ (based on cnot) reverse order
    processor = Processor("SLOS", 6)

    # convert bit to phase
    processor.add((2, 3), BS.H())
    processor.add((4, 5), BS.H())

    # phase flip

    # convert phase to bit
    processor.add((2, 3), BS.H())
    # bit flip
    processor.add((2, 3, 0, 1), catalog['heralded cnot'].build_processor())
    # convert bit to phase
    processor.add((2, 3), BS.H())

    # convert phase to bit
    processor.add((4, 5), BS.H())
    # bit flip
    processor.add((4, 5, 0, 1), catalog['heralded cnot'].build_processor())
    # convert bit to phase
    processor.add((4, 5), BS.H())

    # convert phase to bit
    processor.add((2, 3), BS.H())
    processor.add((4, 5), BS.H())

    measure_processor(processor, b101010)


def test_had_and_cz_rev_cnotbased_post():
    # Testing Had and CZ (based on cnot post) reverse order
    processor = Processor("SLOS", 6)

    # convert bit to phase
    processor.add((2, 3), BS.H())
    processor.add((4, 5), BS.H())

    # phase flip

    # convert phase to bit
    processor.add((0, 1), BS.H())
    # bit flip using CNOT(1,0)
    processor.add((2, 3, 0, 1), catalog['postprocessed cnot'].build_processor())
    processor.clear_postselection()
    # convert bit to phase
    processor.add((0, 1), BS.H())

    # convert phase to bit
    processor.add((0, 1), BS.H())
    # bit flip using CNOT(1,0)
    processor.add((4, 5, 0, 1), catalog['postprocessed cnot'].build_processor())
    processor.clear_postselection()
    # convert bit to phase
    processor.add((0, 1), BS.H())

    # convert phase to bit
    processor.add((2, 3), BS.H())
    processor.add((4, 5), BS.H())

    processor.set_postselection(PostSelect("[0,1]==1 & [2,3]==1 & [4,5]==1"))
    measure_processor(processor, BasicState([1, 0, 1, 0, 1, 0]))
