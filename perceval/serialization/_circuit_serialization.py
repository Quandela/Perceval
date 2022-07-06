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
from perceval.serialization import _schema_circuit_pb2 as pb

from perceval import ACircuit, Circuit
import perceval.lib.phys as phys
import perceval.lib.symb as symb
from perceval.serialization._matrix_serialization import serialize_matrix


class ComponentSerializer:
    def __init__(self):
        self._pb = None

    def serialize(self, r: int, c: ACircuit):
        self._pb = pb.Component()
        self._pb.starting_mode = r
        self._pb.n_mode = c.m
        self._pb.component_type = type(c).__name__
        self._pb.ns = pb.Component.PHYS if '.phys' in c.__module__ else pb.Component.SYMB
        self._serialize(c)
        return self._pb

    @dispatch(phys.BS)
    def _serialize(self, bs: phys.BS):
        pb_bs = pb.BeamSplitterComplex()
        if 'theta' in bs.params:
            pb_bs.theta.serialization = str(bs._theta._value)
        if 'R' in bs.params:
            pb_bs.R.serialization = str(bs._R._value)
        pb_bs.phi_a.serialization = str(bs._phi_a._value)
        pb_bs.phi_b.serialization = str(bs._phi_b._value)
        pb_bs.phi_d.serialization = str(bs._phi_d._value)
        self._pb.beam_splitter_complex.CopyFrom(pb_bs)

    @dispatch(symb.BS)
    def _serialize(self, bs: symb.BS):
        pb_bs = pb.BeamSplitter()
        if 'theta' in bs.params:
            pb_bs.theta.serialization = str(bs._theta._value)
        if 'R' in bs.params:
            pb_bs.R.serialization = str(bs._R._value)
        pb_bs.phi.serialization = str(bs._phi._value)
        self._pb.beam_splitter.CopyFrom(pb_bs)

    @dispatch((phys.PS, symb.PS))
    def _serialize(self, ps):
        pb_ps = pb.PhaseShifter()
        pb_ps.phi.serialization = str(ps._phi._value)
        self._pb.phase_shifter.CopyFrom(pb_ps)

    @dispatch((phys.PERM, symb.PERM))
    def _serialize(self, p):
        pb_perm = pb.Permutation()
        pb_perm.permutations.extend(p.perm_vector)
        self._pb.permutation.CopyFrom(pb_perm)

    @dispatch((phys.Unitary, symb.Unitary))
    def _serialize(self, unitary):
        pb_umat = serialize_matrix(unitary.U)
        pb_unitary = pb.Unitary()
        pb_unitary.mat.CopyFrom(pb_umat)
        if unitary._name != phys.Unitary._name:
            pb_unitary.name = unitary._name
        pb_unitary.use_polarization = unitary.requires_polarization
        self._pb.unitary.CopyFrom(pb_unitary)

    @dispatch((phys.PBS, symb.PBS))
    def _serialize(self, pbs):
        # No need to serialize anything specific for PBS
        pass

    @dispatch((phys.QWP, symb.QWP, phys.HWP, symb.HWP))
    def _serialize(self, wp):
        pb_wp = pb.WavePlate()
        pb_wp.xsi.serialization = str(wp._xsi._value)
        self._pb.wave_plate.CopyFrom(pb_wp)

    @dispatch((phys.WP, symb.WP))
    def _serialize(self, wp):
        pb_wp = pb.WavePlate()
        pb_wp.delta.serialization = str(wp._delta._value)
        pb_wp.xsi.serialization = str(wp._xsi._value)
        self._pb.wave_plate.CopyFrom(pb_wp)

    @dispatch((phys.DT, symb.DT))
    def _serialize(self, dt):
        pb_dt = pb.DT()
        pb_dt.dt.serialization = str(dt._dt._value)
        self._pb.d_t.CopyFrom(pb_dt)

    @dispatch((phys.PR, symb.PR))
    def _serialize(self, pr):
        pb_pr = pb.PolarizationRotator()
        pb_pr.delta.serialization = str(pr._delta._value)
        self._pb.polarization_rotator.CopyFrom(pb_pr)

    @dispatch(Circuit)
    def _serialize(self, circuit: Circuit):
        pb_circ = serialize_circuit(circuit)
        self._pb.circuit.CopyFrom(pb_circ)


def serialize_circuit(circuit: ACircuit) -> pb.Circuit:
    if not isinstance(circuit, Circuit):
        circuit = Circuit(circuit.m).add(circuit)

    pb_circuit = pb.Circuit()
    if circuit._name != Circuit._name:
        pb_circuit.name = circuit._name
    pb_circuit.n_mode = circuit.m
    comp_serializer = ComponentSerializer()
    for r, c in circuit._components:
        pb_circuit.components.extend([comp_serializer.serialize(r[0], c)])
    return pb_circuit
