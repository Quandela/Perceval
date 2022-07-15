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


def test_basic_circuit_h():
    qc = qiskit.QuantumCircuit(1)
    qc.h(0)
    pc = perceval_qiskit.converter(qc, phys)
    c = pc.circuit
    sources = pc._sources
    assert len(sources) == 1
    assert 0 in sources
    assert 1 not in sources
    assert len(c._components) == 1
    assert isinstance(c._components[0][1], phys.Circuit) and len(c._components[0][1]._components) == 1
    c0 = c._components[0][1]._components[0][1]
    assert isinstance(c0, phys.BS)


def test_basic_circuit_s():
    qc = qiskit.QuantumCircuit(1)
    qc.s(0)
    pc = perceval_qiskit.converter(qc, phys)
    c = pc.circuit
    sources = pc._sources
    assert len(sources) == 1
    assert 0 in sources
    assert 1 not in sources
    assert len(c._components) == 1
    assert isinstance(c._components[0][1], phys.Circuit) and len(c._components[0][1]._components) == 1
    r0 = c._components[0][1]._components[0][0]
    c0 = c._components[0][1]._components[0][1]
    assert r0 == (1, )
    assert isinstance(c0, phys.PS)
