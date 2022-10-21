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

from perceval.utils.statevector import BasicState, StateVector, BSSamples
from perceval.utils import simple_float
from ast import literal_eval
import re


def serialize_state(state: BasicState) -> str:
    return str(state)


def deserialize_state(serial_bs) -> BasicState:
    return BasicState(serial_bs)


def deserialize_state_list(states):
    state_list = literal_eval(states)
    return [deserialize_state(s) for s in state_list]


def serialize_statevector(sv: StateVector) -> str:
    sv.normalize()
    ls = []
    for key, value in sv.items():
        real = simple_float(value.real, nsimplify=False)[1]
        imag = simple_float(value.imag, nsimplify=False)[1]
        ls.append("(%s,%s)*%s" % (real, imag, str(key)))
    return "+".join(ls)


def deserialize_statevector(s) -> StateVector:
    sv = StateVector()
    for c in s.split("+"):
        m = re.match(r"\((.*),(.*)\)\*(.*)$", c)
        assert m, "invalid state vector serialization: %s" % s
        sv[BasicState(m.group(3))] = float(m.group(1)) + 1j * float(m.group(2))
    sv._normalized = True
    sv._has_symbolic = False
    return sv


def serialize_bssamples(bss: BSSamples) -> str:
    bs_set = []
    order = []
    for s in bss:
        if s not in bs_set:
            bs_set.append(s)
        order.append(bs_set.index(s))
    return ';'.join([serialize_state(bs) for bs in bs_set]) + '/' + ';'.join([str(i) for i in order])


def deserialize_bssamples(serialized_bss: str) -> BSSamples:
    parts = serialized_bss.split('/')
    assert len(parts) == 2, f"Bad serialized BSSamples: {serialized_bss}"
    bs_set = [deserialize_state(bs) for bs in parts[0].split(';')]
    order = [int(x) for x in parts[1].split(';')]
    output = BSSamples()
    for index in order:
        output.append(bs_set[index])
    return output
