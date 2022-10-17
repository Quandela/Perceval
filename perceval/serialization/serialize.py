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

from multipledispatch import dispatch

from ._matrix_serialization import serialize_matrix
from ._circuit_serialization import serialize_circuit
from perceval.components import ACircuit
from perceval.utils import Matrix, BasicState, SVDistribution, StateVector
from base64 import b64encode


@dispatch(ACircuit)
def serialize(circuit: ACircuit) -> str:
    return ":PCVL:ACircuit:"+b64encode(serialize_circuit(circuit).SerializeToString()).decode('utf-8')


@dispatch(Matrix)
def serialize(m: Matrix) -> str:
    return ":PCVL:Matrix:"+b64encode(serialize_matrix(m).SerializeToString()).decode('utf-8')


@dispatch(BasicState)
def serialize(obj) -> str:
    return ":PCVL:BasicState:"+str(obj)


@dispatch(StateVector)
def serialize(obj) -> str:
    return ":PCVL:StateVector:"+obj.serialize()


@dispatch(SVDistribution)
def serialize(obj) -> str:
    return ":PCVL:SVDistribution:"+str(obj)


@dispatch(dict)
def serialize(obj) -> dict:
    r = {}
    for k, v in obj.items():
        r[serialize(k)] = serialize(v)
    return r


@dispatch(list)
def serialize(obj) -> list:
    r = []
    for k in obj:
        r.append(serialize(k))
    return r

@dispatch(object)
def serialize(obj) -> object:
    return obj


def serialize_to_file(obj, filepath: str) -> None:
    serial_repr = serialize(obj)
    with open(filepath, mode="wb") as f:
        f.write(serial_repr)
