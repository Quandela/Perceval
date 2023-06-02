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

from multipledispatch import dispatch
from functools import wraps
from zlib import compress

from ._matrix_serialization import serialize_matrix
from ._circuit_serialization import serialize_circuit
from ._state_serialization import serialize_state, serialize_statevector, serialize_bssamples
from perceval.components import ACircuit
from perceval.utils import Matrix, BasicState, SVDistribution, BSDistribution, BSCount, BSSamples, StateVector, \
    simple_float
from base64 import b64encode
import json


def to_compress(func):
    @wraps(func)
    def compressor(*args, **kwargs):
        if isinstance(args[0], ACircuit) or kwargs.get('compress', False):
            # ACircuit is always compressed
            serialized_string = func(*args, **kwargs)  # serialized circuit : string format
            serialized_string_compressed = compress(serialized_string.encode('utf-8'))
            # serialized and compressed : byte format
            serialized_string_compressed_byt2str = b64encode(serialized_string_compressed).decode('utf-8')
            # serialized and compressed : string format
            serialized_compressed_byt2str_zip_prefix = ":PCVL:zip:" + serialized_string_compressed_byt2str
            # adding zip prefix
            return serialized_compressed_byt2str_zip_prefix
        else:
            return func(*args, **kwargs)
    return compressor


@to_compress
@dispatch(ACircuit)
def serialize(circuit: ACircuit, compress=True) -> str:
    return ":PCVL:ACircuit:" + b64encode(serialize_circuit(circuit).SerializeToString()).decode('utf-8')


@to_compress
@dispatch(Matrix)
def serialize(m: Matrix, compress=False) -> str:
    return ":PCVL:Matrix:" + b64encode(serialize_matrix(m).SerializeToString()).decode('utf-8')


@to_compress
@dispatch(BasicState)
def serialize(obj, compress=False) -> str:
    return ":PCVL:BasicState:" + serialize_state(obj)


@to_compress
@dispatch(StateVector)
def serialize(sv, compress=False) -> str:
    return ":PCVL:StateVector:" + serialize_statevector(sv)


@to_compress
@dispatch(SVDistribution)
def serialize(dist, compress=False) -> str:
    return ":PCVL:SVDistribution:{" \
           + ";".join(["%s=%s" % (serialize_statevector(k), simple_float(v, nsimplify=False)[1])
                       for k, v in dist.items()]) \
           + "}"


@to_compress
@dispatch(BSDistribution)
def serialize(dist, compress=False) -> str:
    return ":PCVL:BSDistribution:{" \
           + ";".join(["%s=%s" % (serialize_state(k), simple_float(v, nsimplify=False)[1]) for k, v in dist.items()]) \
           + "}"


@to_compress
@dispatch(BSCount)
def serialize(obj, compress=False) -> str:
    return ":PCVL:BSCount:{" \
           + ";".join(["%s=%s" % (serialize_state(k), str(v)) for k, v in obj.items()]) \
           + "}"


@to_compress
@dispatch(BSSamples)
def serialize(obj, compress=False) -> str:
    return ":PCVL:BSSamples:" + serialize_bssamples(obj)


@to_compress
@dispatch(dict)
def serialize(obj, compress=False) -> dict:
    r = {}
    for k, v in obj.items():
        r[serialize(k)] = serialize(v)
    return r


@to_compress
@dispatch(list)
def serialize(obj, compress=False) -> list:
    r = []
    for k in obj:
        r.append(serialize(k))
    return r


@to_compress
@dispatch(object)
def serialize(obj, compress=False) -> object:
    return obj


def serialize_to_file(obj, filepath: str, compress=False) -> None:
    serial_repr = serialize(obj)
    with open(filepath, mode="w") as f:
        f.write(json.dumps(serial_repr))
