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
from ._state_serialization import serialize_state, serialize_statevector, serialize_bssamples
from perceval.components import ACircuit
from perceval.utils import Matrix, BasicState, SVDistribution, BSDistribution, BSCount, BSSamples, StateVector, \
    simple_float
from base64 import b64encode


@dispatch(ACircuit)
def serialize(circuit: ACircuit) -> str:
    return ":PCVL:ACircuit:" + b64encode(serialize_circuit(circuit).SerializeToString()).decode('utf-8')


@dispatch(Matrix)
def serialize(m: Matrix) -> str:
    return ":PCVL:Matrix:" + b64encode(serialize_matrix(m).SerializeToString()).decode('utf-8')


@dispatch(BasicState)
def serialize(obj) -> str:
    return ":PCVL:BasicState:" + serialize_state(obj)


@dispatch(StateVector)
def serialize(sv) -> str:
    return ":PCVL:StateVector:" + serialize_statevector(sv)


@dispatch(SVDistribution)
def serialize(dist) -> str:
    return ":PCVL:SVDistribution:{" \
           + ";".join(["%s=%s" % (serialize_statevector(k), simple_float(v, nsimplify=False)[1])
                       for k, v in dist.items()]) \
           + "}"


@dispatch(BSDistribution)
def serialize(dist) -> str:
    return ":PCVL:BSDistribution:{" \
           + ";".join(["%s=%s" % (serialize_state(k), simple_float(v, nsimplify=False)[1]) for k, v in dist.items()]) \
           + "}"


@dispatch(BSCount)
def serialize(obj) -> str:
    return ":PCVL:BSCount:{" \
           + ";".join(["%s=%s" % (serialize_state(k), str(v)) for k, v in obj.items()]) \
           + "}"


@dispatch(BSSamples)
def serialize(obj) -> str:
    return ":PCVL:BSSamples:" + serialize_bssamples(obj)


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
