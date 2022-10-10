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
from ._fockstate_serialization import serialize_state
from perceval.components import ACircuit
from perceval.utils import Matrix, BasicState, SVDistribution, StateVector
from base64 import b64encode


@dispatch(ACircuit)
def serialize(circuit: ACircuit) -> str:
    return serialize_circuit(circuit).SerializeToString()


@dispatch(Matrix)
def serialize(m: Matrix) -> str:
    return serialize_matrix(m).SerializeToString()


@dispatch(BasicState)
def serialize(obj) -> str:
    return str(obj)


@dispatch(StateVector)
def serialize(obj) -> str:
    return str(obj)


@dispatch(SVDistribution)
def serialize(obj) -> str:
    return str(obj)

@dispatch(dict)
def serialize(obj):
    print(type(obj), obj)
    r = {}
    for k, v in obj.items():
        r[serialize(k)] = serialize(v)
    return r


@dispatch(list)
def serialize(obj) -> str:
    r = []
    for v in obj:
        r.append(serialize(v))
    return r


@dispatch(int)
def serialize(obj):
    return obj


@dispatch(float)
def serialize(obj):
    return obj


@dispatch(complex)
def serialize(obj):
    return obj


@dispatch(object)
def serialize(obj) -> str:
    return str(obj)


def serialize_to_file(obj, filepath: str) -> None:
    serial_repr = serialize(obj)
    with open(filepath, mode="wb") as f:
        f.write(serial_repr)


def bytes_to_jsonstring(var):
    return b64encode(var).decode('utf-8')
