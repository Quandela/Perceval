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

from os import path
from inspect import signature

from perceval.components import Circuit
from perceval.utils import Matrix
from perceval.serialization import _matrix_serialization
import perceval.serialization._component_deserialization as _cd
from perceval.serialization import _schema_circuit_pb2 as pb


def deserialize_matrix(pb_mat: str | bytes | pb.Matrix) -> Matrix:
    if not isinstance(pb_mat, pb.Matrix):
        pb_binary_repr = pb_mat
        pb_mat = pb.Matrix()
        pb_mat.ParseFromString(pb_binary_repr)
    return _matrix_serialization.deserialize_pb_matrix(pb_mat)


def matrix_from_file(filepath: str) -> Matrix:
    if not path.isfile(filepath):
        raise FileNotFoundError(f'No file at path {filepath}')
    with open(filepath, 'rb') as f:
        return deserialize_matrix(f.read())


def deserialize_circuit(pb_circ: str | bytes | pb.Circuit) -> Circuit:
    if not isinstance(pb_circ, pb.Circuit):
        pb_binary_repr = pb_circ
        pb_circ = pb.Circuit()
        pb_circ.ParseFromString(pb_binary_repr)
    builder = CircuitBuilder(pb_circ.n_mode, pb_circ.name)
    for pb_c in pb_circ.components:
        builder.add(pb_c)
    return builder.retrieve()


def circuit_from_file(filepath: str) -> Circuit:
    if not path.isfile(filepath):
        raise FileNotFoundError(f'No file at path {filepath}')
    with open(filepath, 'rb') as f:
        return deserialize_circuit(f.read())


class CircuitBuilder:

    deserialize_fn = {
        'circuit': deserialize_circuit,
        'beam_splitter': _cd.deserialize_symb_bs,
        'beam_splitter_complex': _cd.deserialize_phys_bs,
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
            if len(signature(func).parameters) == 1:
                component = func(serial_sub_comp)
            else:
                component = func(serial_sub_comp, serial_comp.ns)

        if component is None:
            raise NotImplementedError(f'Component could not be deserialized (type = {t})')
        self._circuit.add(serial_comp.starting_mode, component, merge=False)

    def retrieve(self):
        return self._circuit
