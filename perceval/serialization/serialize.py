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
from ._detector_serialization import serialize_bs_layer, serialize_detector
from ._experiment_serialization import serialize_experiment
from ._matrix_serialization import serialize_matrix
from ._circuit_serialization import serialize_circuit, serialize_component, serialize_herald, serialize_port
from ._state_serialization import serialize_state, serialize_statevector, serialize_bssamples
from perceval.components import ACircuit, BSLayeredPPNR, Detector, AComponent, Herald, Port, Experiment
from perceval.utils import Matrix, BasicState, SVDistribution, BSDistribution, BSCount, BSSamples, StateVector, \
    simple_float, NoiseModel, PostSelect
from base64 import b64encode
import json


def b64encoding(obj: bytes) -> str:
    return b64encode(obj).decode('utf-8')


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
    serialized_string_compressed_byt2str = b64encoding(serialized_string_compressed)  # base64 to string
    return ZIP_PREFIX + serialized_string_compressed_byt2str


@dispatch(AComponent, compress=(list, bool))
def serialize(component: AComponent, compress=None) -> str:
    if compress is None:
        compress = True
    tag = COMPONENT_TAG
    compress = _handle_compress_parameter(compress, tag)
    return _handle_compression(
        f"{PCVL_PREFIX}{tag}{SEP}" + b64encoding(serialize_component(component).SerializeToString()),
        do_compress=compress)


@dispatch(ACircuit, compress=(list, bool))
def serialize(circuit: ACircuit, compress=None) -> str:
    if compress is None:
        compress = True
    tag = CIRCUIT_TAG
    compress = _handle_compress_parameter(compress, tag)
    return _handle_compression(
        f"{PCVL_PREFIX}{tag}{SEP}" + b64encoding(serialize_circuit(circuit).SerializeToString()),
        do_compress=compress)


@dispatch(Experiment, compress=(list, bool))
def serialize(experiment: Experiment, compress=None) -> str:
    if compress is None:
        compress = True
    tag = EXPERIMENT_TAG
    compress = _handle_compress_parameter(compress, tag)
    return _handle_compression(
        f"{PCVL_PREFIX}{tag}{SEP}" + b64encoding(serialize_experiment(experiment).SerializeToString()),
        do_compress=compress)


@dispatch(Herald, compress=(list, bool))
def serialize(herald: Herald, compress=None) -> str:
    if compress is None:
        compress = True
    tag = HERALD_TAG
    compress = _handle_compress_parameter(compress, tag)
    return _handle_compression(
        f"{PCVL_PREFIX}{tag}{SEP}" + b64encoding(serialize_herald(herald).SerializeToString()), do_compress=compress)


@dispatch(Port, compress=(list, bool))
def serialize(port: Port, compress=None) -> str:
    if compress is None:
        compress = True
    tag = PORT_TAG
    compress = _handle_compress_parameter(compress, tag)
    return _handle_compression(
        f"{PCVL_PREFIX}{tag}{SEP}" + b64encoding(serialize_port(port).SerializeToString()), do_compress=compress)


@dispatch(Matrix, compress=(list, bool))
def serialize(m: Matrix, compress=None) -> str:
    if compress is None:
        compress = False
    tag = MATRIX_TAG
    compress = _handle_compress_parameter(compress, tag)
    return _handle_compression(f"{PCVL_PREFIX}{tag}{SEP}" + b64encoding(serialize_matrix(m).SerializeToString()),
                               do_compress=compress)


@dispatch(BasicState, compress=(list, bool))
def serialize(obj, compress=None) -> str:
    if compress is None:
        compress = False
    tag = BS_TAG
    compress = _handle_compress_parameter(compress, tag)
    return _handle_compression(f"{PCVL_PREFIX}{tag}{SEP}" + serialize_state(obj), do_compress=compress)



@dispatch(StateVector, compress=(list, bool))
def serialize(sv, compress=None) -> str:
    if compress is None:
        compress = False
    tag = SV_TAG
    compress = _handle_compress_parameter(compress, tag)
    return _handle_compression(f"{PCVL_PREFIX}{tag}{SEP}" + serialize_statevector(sv), do_compress=compress)


@dispatch(SVDistribution, compress=(list, bool))
def serialize(dist: SVDistribution, compress=None) -> str:
    if compress is None:
        compress = False
    tag = SVD_TAG
    compress = _handle_compress_parameter(compress, tag)
    serial_svd = f"{PCVL_PREFIX}{tag}{SEP}{{" \
           + ";".join(["%s=%s" % (serialize_statevector(k), simple_float(v, nsimplify=False)[1])
                       for k, v in dist.items()]) \
           + "}"
    return _handle_compression(serial_svd, do_compress=compress)


@dispatch(BSDistribution, compress=(list, bool))
def serialize(dist: BSDistribution, compress=None) -> str:
    if compress is None:
        compress = True
    tag = BSD_TAG
    compress = _handle_compress_parameter(compress, tag)
    serial_bsd = f"{PCVL_PREFIX}{tag}{SEP}{{" \
           + ";".join(["%s=%s" % (serialize_state(k), simple_float(v, nsimplify=False)[1]) for k, v in dist.items()]) \
           + "}"
    return _handle_compression(serial_bsd, do_compress=compress)


@dispatch(BSCount, compress=(list, bool))
def serialize(obj, compress=None) -> str:
    if compress is None:
        compress = True
    tag = BSC_TAG
    compress = _handle_compress_parameter(compress, tag)
    serial_bsc = f"{PCVL_PREFIX}{tag}{SEP}{{" \
           + ";".join(["%s=%s" % (serialize_state(k), str(v)) for k, v in obj.items()]) \
           + "}"
    return _handle_compression(serial_bsc, do_compress=compress)


@dispatch(BSSamples, compress=(list, bool))
def serialize(obj, compress=None) -> str:
    if compress is None:
        compress = True
    tag = BSS_TAG
    compress = _handle_compress_parameter(compress, tag)
    return _handle_compression(f"{PCVL_PREFIX}{tag}{SEP}" + serialize_bssamples(obj), do_compress=compress)


@dispatch(NoiseModel, compress=(list, bool))
def serialize(obj, compress=None):
    if compress is None:
        compress = False
    tag = NOISE_TAG
    compress = _handle_compress_parameter(compress, tag)
    return _handle_compression(f"{PCVL_PREFIX}{tag}{SEP}" + json.dumps(obj.__dict__()), do_compress=compress)


@dispatch(PostSelect, compress=(list, bool))
def serialize(ps: PostSelect, compress=None):
    if compress is None:
        compress = False
    tag = POSTSELECT_TAG
    compress = _handle_compress_parameter(compress, tag)
    return _handle_compression(f"{PCVL_PREFIX}{tag}{SEP}{ps}", do_compress=compress)


@dispatch(BSLayeredPPNR, compress=(list, bool))
def serialize(obj: BSLayeredPPNR, compress=None):
    if compress is None:
        compress = False
    tag = BS_LAYERED_DETECTOR_TAG
    compress = _handle_compress_parameter(compress, tag)
    return _handle_compression(f"{PCVL_PREFIX}{tag}{SEP}" + b64encoding(serialize_bs_layer(obj).SerializeToString()),
                               do_compress=compress)


@dispatch(Detector, compress=(list, bool))
def serialize(obj: Detector, compress=None):
    if compress is None:
        compress = False
    tag = DETECTOR_TAG
    compress = _handle_compress_parameter(compress, tag)
    return _handle_compression(f"{PCVL_PREFIX}{tag}{SEP}" + b64encoding(serialize_detector(obj).SerializeToString()),
                               do_compress=compress)


@dispatch(dict, compress=(list, bool))
def serialize(obj, compress=None) -> dict:
    r = {}
    for k, v in obj.items():
        r[serialize(k, compress=compress)] = serialize(v, compress=compress)
    return r


@dispatch(list, compress=(list, bool))
def serialize(obj, compress=None) -> list:
    r = []
    for k in obj:
        r.append(serialize(k, compress=compress))
    return r


@dispatch(object, compress=(list, bool))
def serialize(obj, compress=None) -> object:
    return obj


def serialize_to_file(obj, filepath: str, compress=None) -> None:
    serial_repr = serialize(obj, compress=compress)
    with open(filepath, mode="w") as f:
        f.write(json.dumps(serial_repr))
