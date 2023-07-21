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

try:
    from qat.lang.AQASM import Program, H, CNOT
except ModuleNotFoundError as e:
    pytest.skip("need `myqlm` module", allow_module_level=True)

from perceval import BasicState, StateVector, Circuit
from perceval.converters import MyQLMConverter
import perceval.components.unitary_components as comp
from perceval.components import catalog

# todo : implement further assertions


def test_basic_circuit_h():
    convertor = MyQLMConverter(catalog)
    qprog = Program()  # Create a Program
    qbits = qprog.qalloc(1)  # Allocate some qbits
    qprog.apply(H, qbits[0])  # Apply H gate
    myqlmc = qprog.to_circ()  # Export this program into a quantum circuit

    pc = convertor.convert(myqlmc)

    c = pc.linear_circuit()
    assert c.m == 2 * len(qbits)


def test_cnot_1_heralded():
    convertor = MyQLMConverter(catalog)
    qprog = Program()
    qbits = qprog.qalloc(2)  # AllocateS 2 qbits
    qprog.apply(CNOT, qbits[0], qbits[1])
    myqlmc = qprog.to_circ()

    pc = convertor.convert(myqlmc, use_postselection=False)
    assert pc.circuit_size == 8
    assert pc.m == 4


def test_cnot_H():
    convertor = MyQLMConverter(catalog)
    qprog = Program()
    qbits = qprog.qalloc(2)  # AllocateS 2 qbits
    qprog.apply(H, qbits[0])
    qprog.apply(CNOT, qbits[0], qbits[1])
    qprog.apply(H, qbits[1])
    myqlmc = qprog.to_circ()

    pc = convertor.convert(myqlmc, use_postselection=False)
    assert pc.circuit_size == 8
    assert pc.m == 4
