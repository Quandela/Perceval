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

import copy
from packaging.version import Version
from perceval.components.compiled_circuit import ChipParameters, CompiledCircuit
from perceval.utils.parameter import Parameter
import pytest

def test_need_compilation():
    chip_origin = ChipParameters("myChip", [Parameter("phi_1"), Parameter("phi_2"), Parameter("phi_3")], Version("1.0"))

    chip_newer = ChipParameters("myChip", [Parameter("phi_1"), Parameter("phi_2"), Parameter("phi_3")], Version("2.0"))
    assert not chip_origin.need_compilation(chip_origin)
    assert chip_origin.need_compilation(chip_newer)

def test_parameters():
    chip_specifications = ChipParameters("myChip", [Parameter("phi_1"), Parameter("phi_2"), Parameter("phi_3")])
    circuit = CompiledCircuit(2, chip_specifications)
    assert len(circuit.get_parameters()) == 3
    assert not any(p.defined for p in circuit.get_parameters())

    # Have compilation being executed somewhere
    compilation_parameters = copy.deepcopy(chip_specifications)
    assert compilation_parameters.get_values() == None
    compilation_parameters.set_values([1., 3., 0.5])
    assert compilation_parameters.get_values() == [1., 3., 0.5]

    circuit.setParams(compilation_parameters)
    assert all(p.defined for p in circuit.get_parameters())
    assert circuit.param("phi_1").evalf() == 1.
    assert circuit.param("phi_2").evalf() == 3.
    assert circuit.param("phi_3").evalf() == 0.5
