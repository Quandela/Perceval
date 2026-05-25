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

from perceval import Computation, CommandFactory, Experiment


def test_parameters():
    experiment = Experiment()  # This Experiment does not have anything here
    comp = Computation(CommandFactory.sample_count, experiment)

    with pytest.raises(ValueError):
        comp.validate()

    # Assume Command is well tested - no need to test its internal behaviour
    comp.add_params(10_000)
    assert comp.parameters == {"max_samples": 10_000}

    comp.add_params(max_shots=20_000)
    assert comp.parameters == {"max_samples": 10_000, "max_shots": 20_000}

    comp.add_params(max_shots=100_000)
    assert comp.parameters == {"max_samples": 10_000, "max_shots": 100_000}

    comp.add_params(max_samples=50_000)
    assert comp.parameters == {"max_samples": 50_000, "max_shots": 100_000}

    comp.validate()
