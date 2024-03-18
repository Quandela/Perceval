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
from zlib import compress as zlib_compress

from ._constants import *
from ._matrix_serialization import serialize_matrix
from ._circuit_serialization import serialize_circuit
from ._state_serialization import serialize_state, serialize_statevector, serialize_bssamples
from perceval.components import ACircuit
from perceval.utils import Matrix, BasicState, SVDistribution, BSDistribution, BSCount, BSSamples, StateVector, \
    simple_float, NoiseModel, PostSelect
from base64 import b64encode
import json


@dispatch(bool, str)
def _handle_compress_parameter(compress, type_str) -> bool:
    return compress


@dispatch(list, str)
def _handle_compress_parameter(compress, type_str) -> bool:
    return type_str in compress


def _handle_compression(serialized_obj: str, do_compress: bool) -> str:
    if not do_compress:
        return serialized_obj
    serialized_string_compressed = zlib_compress(serialized_obj.encode('utf-8'))  # Compress byte to byte
    serialized_string_compressed_byt2str = b64encode(serialized_string_compressed).decode('utf-8')  # base64 to string
    return ZIP_PREFIX + serialized_string_compressed_byt2str


@dispatch(ACircuit, compress=(list, bool))
def serialize(circuit: ACircuit, compress=True) -> str:
    tag = CIRCUIT_TAG
    compress = _handle_compress_parameter(compress, tag)
    return _handle_compression(
        f"{PCVL_PREFIX}{tag}{SEP}" + b64encode(serialize_circuit(circuit).SerializeToString()).decode('utf-8'),
        do_compress=compress)


@dispatch(Matrix, compress=(list, bool))
def serialize(m: Matrix, compress=False) -> str:
    tag = MATRIX_TAG
    compress = _handle_compress_parameter(compress, tag)
    return _handle_compression(f"{PCVL_PREFIX}{tag}{SEP}" + b64encode(serialize_matrix(m).SerializeToString()).decode('utf-8'),
                               do_compress=compress)


@dispatch(BasicState, compress=(list, bool))
def serialize(obj, compress=False) -> str:
    tag = BS_TAG
    compress = _handle_compress_parameter(compress, tag)
    return _handle_compression(f"{PCVL_PREFIX}{tag}{SEP}" + serialize_state(obj), do_compress=compress)



@dispatch(StateVector, compress=(list, bool))
def serialize(sv, compress=False) -> str:
    tag = SV_TAG
    compress = _handle_compress_parameter(compress, tag)
    return _handle_compression(f"{PCVL_PREFIX}{tag}{SEP}" + serialize_statevector(sv), do_compress=compress)


@dispatch(SVDistribution, compress=(list, bool))
def serialize(dist: SVDistribution, compress=False) -> str:
    tag = SVD_TAG
    compress = _handle_compress_parameter(compress, tag)
    serial_svd = f"{PCVL_PREFIX}{tag}{SEP}{{" \
           + ";".join(["%s=%s" % (serialize_statevector(k), simple_float(v, nsimplify=False)[1])
                       for k, v in dist.items()]) \
           + "}"
    return _handle_compression(serial_svd, do_compress=compress)


@dispatch(BSDistribution, compress=(list, bool))
def serialize(dist: BSDistribution, compress=True) -> str:
    tag = BSD_TAG
    compress = _handle_compress_parameter(compress, tag)
    serial_bsd = f"{PCVL_PREFIX}{tag}{SEP}{{" \
           + ";".join(["%s=%s" % (serialize_state(k), simple_float(v, nsimplify=False)[1]) for k, v in dist.items()]) \
           + "}"
    return _handle_compression(serial_bsd, do_compress=compress)


@dispatch(BSCount, compress=(list, bool))
def serialize(obj, compress=True) -> str:
    tag = BSC_TAG
    compress = _handle_compress_parameter(compress, tag)
    serial_bsc = f"{PCVL_PREFIX}{tag}{SEP}{{" \
           + ";".join(["%s=%s" % (serialize_state(k), str(v)) for k, v in obj.items()]) \
           + "}"
    return _handle_compression(serial_bsc, do_compress=compress)


@dispatch(BSSamples, compress=(list, bool))
def serialize(obj, compress=True) -> str:
    tag = BSS_TAG
    compress = _handle_compress_parameter(compress, tag)
    return _handle_compression(f"{PCVL_PREFIX}{tag}{SEP}" + serialize_bssamples(obj), do_compress=compress)


@dispatch(NoiseModel, compress=(list, bool))
def serialize(obj, compress=False):
    tag = NOISE_TAG
    compress = _handle_compress_parameter(compress, tag)
    return _handle_compression(f"{PCVL_PREFIX}{tag}{SEP}" + json.dumps(obj.__dict__()), do_compress=compress)


@dispatch(PostSelect, compress=(list, bool))
def serialize(ps: PostSelect, compress=False):
    tag = POSTSELECT_TAG
    compress = _handle_compress_parameter(compress, tag)
    return _handle_compression(f"{PCVL_PREFIX}{tag}{SEP}{ps}", do_compress=compress)


@dispatch(dict, compress=(list, bool))
def serialize(obj, compress=False) -> dict:
    r = {}
    for k, v in obj.items():
        r[serialize(k, compress=compress)] = serialize(v, compress=compress)
    return r


@dispatch(list, compress=(list, bool))
def serialize(obj, compress=False) -> list:
    r = []
    for k in obj:
        r.append(serialize(k, compress=compress))
    return r


@dispatch(object, compress=(list, bool))
def serialize(obj, compress=False) -> object:
    return obj


def serialize_to_file(obj, filepath: str, compress=False) -> None:
    serial_repr = serialize(obj, compress=compress)
    with open(filepath, mode="w") as f:
        f.write(json.dumps(serial_repr))
