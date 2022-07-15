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
import pytest

try:
    import qiskit
except Exception as e:
    pytest.skip("need `qiskit` module", allow_module_level=True)

from perceval.converters import perceval_qiskit
import perceval.lib.phys as phys

def test_basiccircuit_h():
    qc = qiskit.QuantumCircuit(1)
    qc.h(0)
    pc = perceval_qiskit.converter(qc, phys)
    pcvl.pdisplay(pc, recursive=True)


def test_basic_circuit_s():
    qc = qiskit.QuantumCircuit(1)
    qc.s(0)
    pc = perceval_qiskit.converter(qc, phys)
    descr_pc = pc.describe()
    print(descr_pc)
