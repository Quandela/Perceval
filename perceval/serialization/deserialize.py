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
import json
from os import path
from typing import Union

from perceval.components import Circuit
from perceval.utils import Matrix, BSDistribution, SVDistribution, BasicState, BSCount
from perceval.serialization import _matrix_serialization, deserialize_state
from ._state_serialization import deserialize_statevector, deserialize_bssamples
import perceval.serialization._component_deserialization as _cd
from perceval.serialization import _schema_circuit_pb2 as pb
from base64 import b64decode


def deserialize_float(floatstring):
    return float(floatstring)


def deserialize_matrix(pb_mat: Union[str, pb.Matrix]) -> Matrix:
    if not isinstance(pb_mat, pb.Matrix):
        pb_binary_repr = pb_mat
        pb_mat = pb.Matrix()
        assert pb_binary_repr.startswith(":PCVL:Matrix:")
        pb_mat.ParseFromString(b64decode(pb_binary_repr[13:]))
    return _matrix_serialization.deserialize_pb_matrix(pb_mat)


def matrix_from_file(filepath: str) -> Matrix:
    """
    Deserialize a matrix from a binary file
    """
    if not path.isfile(filepath):
        raise FileNotFoundError(f'No file at path {filepath}')
    with open(filepath, 'rb') as f:
        return deserialize_matrix(f.read())


def deserialize_circuit(pb_circ: Union[str, pb.Circuit]) -> Circuit:
    if not isinstance(pb_circ, pb.Circuit):
        pb_binary_repr = pb_circ
        pb_circ = pb.Circuit()
        assert pb_binary_repr.startswith(":PCVL:ACircuit:")
        pb_circ.ParseFromString(b64decode(pb_binary_repr[15:]))
    builder = CircuitBuilder(pb_circ.n_mode, pb_circ.name)
    for pb_c in pb_circ.components:
        builder.add(pb_c)
    return builder.retrieve()


def circuit_from_file(filepath: str) -> Circuit:
    """
    Deserialize a circuit from a binary file
    """
    if not path.isfile(filepath):
        raise FileNotFoundError(f'No file at path {filepath}')
    with open(filepath, 'rb') as f:
        return deserialize_circuit(f.read())


# TODO remove ?
# def deserialize_sample_count(count: dict) -> dict:
#     count = {deserialize_state(state): ct for state, ct in count.items()}
#     return count


def deserialize_svdistribution(serial_svd):
    assert serial_svd[0] == '{' and serial_svd[-1] == '}', "Invalid serialized SVDistribution"
    svd = SVDistribution()
    for s in serial_svd[1:-1].split(";"):
        k, v = s.split("=")
        svd[deserialize_statevector(k)] = float(v)
    return svd


def deserialize_bsdistribution(serial_bsd):
    assert serial_bsd[0] == '{' and serial_bsd[-1] == '}', "Invalid serialized BSDistribution"
    bsd = BSDistribution()
    for s in serial_bsd[1:-1].split(";"):
        k, v = s.split("=")
        bsd[deserialize_state(k)] = float(v)
    return bsd


def deserialize_bscount(serial_bsc):
    assert serial_bsc[0] == '{' and serial_bsc[-1] == '}', "Invalid serialized BSCount"
    bsc = BSCount()
    for s in serial_bsc[1:-1].split(";"):
        k, v = s.split("=")
        bsc[deserialize_state(k)] = int(v)
    return bsc


def deserialize(obj):
    if isinstance(obj, dict):
        r = {}
        for k, v in obj.items():
            r[deserialize(k)] = deserialize(v)
    elif isinstance(obj, list):
        r = []
        for k in obj:
            r.append(deserialize(k))
    elif isinstance(obj, str) and obj.startswith(":PCVL:"):
        p = obj[6:].find(":")
        cl = obj[6:p+6]
        sobj = obj[p+7:]
        if cl == "BasicState":
            r = BasicState(sobj)
        elif cl == "StateVector":
            r = deserialize_statevector(sobj)
        elif cl == "SVDistribution":
            r = deserialize_svdistribution(sobj)
        elif cl == "BSDistribution":
            r = deserialize_bsdistribution(sobj)
        elif cl == "BSCount":
            r = deserialize_bscount(sobj)
        elif cl == "BSSamples":
            r = deserialize_bssamples(sobj)
        elif cl == "Matrix":
            r = deserialize_matrix(obj)
        elif cl == "ACircuit":
            r = deserialize_circuit(obj)
        else:
            raise NotImplementedError(f"No deserializer found for {cl}")
    else:
        r = obj
    return r


def deserialize_file(filepath: str):
    """
    Agnosticly deserialize any supported type from a text file.
    """
    if not path.isfile(filepath):
        raise FileNotFoundError(f'No file at path {filepath}')
    with open(filepath, 'r') as f:
        return deserialize(f.read())

# TODO remove ?
# def sample_count_from_file(filepath: str) -> dict:
#     if not path.isfile(filepath):
#         raise FileNotFoundError(f'No file at path {filepath}')
#     with open(filepath, 'rb') as f:
#         return deserialize_sample_count(f.read())


class CircuitBuilder:
    deserialize_fn = {
        'circuit': deserialize_circuit,
        'beam_splitter': _cd.deserialize_bs,
        'phase_shifter': _cd.deserialize_ps,
        'permutation': _cd.deserialize_perm,
        'unitary': _cd.deserialize_unitary,
        'wave_plate': _cd.deserialize_wp,
        'quarter_wave_plate': _cd.deserialize_qwp,
        'half_wave_plate': _cd.deserialize_hwp,
        'time_delay': _cd.deserialize_dt,
        'polarization_rotator': _cd.deserialize_pr,
        'polarized_beam_splitter': _cd.deserialize_pbs
    }

    def __init__(self, m: int, name: str):
        if not name:
            name = None
        self._circuit = Circuit(m=m, name=name)

    def add(self, serial_comp):
        component = None
        t = serial_comp.WhichOneof('type')
        serial_sub_comp = getattr(serial_comp, t)
        # find the correct deserialization function and use it
        if t in CircuitBuilder.deserialize_fn:
            func = CircuitBuilder.deserialize_fn[t]
            component = func(serial_sub_comp)

        if component is None:
            raise NotImplementedError(f'Component could not be deserialized (type = {t})')
        self._circuit.add(serial_comp.starting_mode, component, merge=False)

    def retrieve(self):
        return self._circuit
