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

from perceval import Circuit, Matrix
import perceval.lib.phys as phys
import perceval.lib.symb as symb
from perceval.serialization import _matrix_serialization, _expr_serialization as _expr
from perceval.serialization import _schema_circuit_pb2 as pb


class CircuitBuilder:

    def __init__(self, m: int, name: str):
        if not name:
            name = None
        self._circuit = Circuit(m=m, name=name)
        self._pb_comp = []

    def add(self, serial_comp):
        component = None
        if serial_comp.HasField('circuit'):
            component = deserialize_circuit(serial_comp.circuit)
        elif serial_comp.HasField('beam_splitter'):
            component = self._deserialize_symb_bs(serial_comp.beam_splitter)
        elif serial_comp.HasField('beam_splitter_complex'):
            component = self._deserialize_phys_bs(serial_comp.beam_splitter_complex)
        elif serial_comp.HasField('phase_shifter'):
            component = self._deserialize_ps(serial_comp.phase_shifter, serial_comp.ns)
        elif serial_comp.HasField('permutation'):
            component = self._deserialize_perm(serial_comp.permutation, serial_comp.ns)
        elif serial_comp.HasField('unitary'):
            component = self._deserialize_unitary(serial_comp.unitary, serial_comp.ns)
        elif serial_comp.HasField('wave_plate'):
            if serial_comp.component_type == 'WP':
                component = self._deserialize_wp(serial_comp.wave_plate, serial_comp.ns)
            elif serial_comp.component_type == 'QWP':
                component = self._deserialize_qwp(serial_comp.wave_plate, serial_comp.ns)
            elif serial_comp.component_type == 'HWP':
                component = self._deserialize_hwp(serial_comp.wave_plate, serial_comp.ns)
        elif serial_comp.HasField('d_t'):
            component = self._deserialize_dt(serial_comp.d_t, serial_comp.ns)
        elif serial_comp.component_type == 'PBS':
            component = phys.PBS() if serial_comp.ns == pb.Component.PHYS else symb.PBS()

        if component is None:
            raise NotImplementedError('Component could not be deserialized')
        self._circuit.add(serial_comp.starting_mode, component, merge=False)

    def _deserialize_ps(self, serial_ps: pb.PhaseShifter, ns: int):
        return_type = phys.PS if ns == pb.Component.PHYS else symb.PS
        return return_type(_expr.deserialize_expr(serial_ps.phi))

    def _deserialize_phys_bs(self, serial_bs: pb.BeamSplitterComplex) -> phys.BS:
        args = {}
        if serial_bs.HasField('R'):
            args['R'] = _expr.deserialize_expr(serial_bs.R)
        if serial_bs.HasField('theta'):
            args['theta'] = _expr.deserialize_expr(serial_bs.theta)
        args['phi_a'] = _expr.deserialize_expr(serial_bs.phi_a)
        args['phi_b'] = _expr.deserialize_expr(serial_bs.phi_b)
        args['phi_d'] = _expr.deserialize_expr(serial_bs.phi_d)
        return phys.BS(**args)

    def _deserialize_perm(self, serial_perm, ns: int):
        return_type = phys.PERM if ns == pb.Component.PHYS else symb.PERM
        return return_type([x for x in serial_perm.permutations])

    def _deserialize_unitary(self, serial_unitary, ns: int):
        return_type = phys.Unitary if ns == pb.Component.PHYS else symb.Unitary
        m = deserialize_matrix(serial_unitary.mat)
        return return_type(U=m)

    def _deserialize_symb_bs(self, serial_bs: pb.BeamSplitter) -> symb.BS:
        args = {}
        if serial_bs.HasField('R'):
            args['R'] = _expr.deserialize_expr(serial_bs.R)
        if serial_bs.HasField('theta'):
            args['theta'] = _expr.deserialize_expr(serial_bs.theta)
        return symb.BS(**args)

    def retrieve(self):
        return self._circuit

    def _deserialize_wp(self, serial_wp, ns):
        return_type = phys.WP if ns == pb.Component.PHYS else symb.WP
        return return_type(_expr.deserialize_expr(serial_wp.delta), _expr.deserialize_expr(serial_wp.xsi))

    def _deserialize_qwp(self, serial_qwp, ns):
        return_type = phys.QWP if ns == pb.Component.PHYS else symb.QWP
        return return_type(_expr.deserialize_expr(serial_qwp.xsi))

    def _deserialize_hwp(self, serial_hwp, ns):
        return_type = phys.HWP if ns == pb.Component.PHYS else symb.HWP
        return return_type(_expr.deserialize_expr(serial_hwp.xsi))

    def _deserialize_dt(self, serial_dt, ns):
        return_type = phys.DT if ns == pb.Component.PHYS else symb.DT
        return return_type(_expr.deserialize_expr(serial_dt.dt))


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
