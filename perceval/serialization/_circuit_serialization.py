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

from perceval.serialization import _schema_circuit_pb2 as pb
from perceval.components import ACircuit, Circuit, AComponent, Herald, Port
import perceval.components.unitary_components as comp
import perceval.components.non_unitary_components as nu
from perceval.components import FFConfigurator, FFCircuitProvider
from perceval.serialization._matrix_serialization import serialize_matrix
from perceval.serialization._parameter_serialization import serialize_parameter


class ComponentSerializer:
    def __init__(self):
        self._pb = None

    def serialize(self, r: int, c: AComponent):
        self._pb = pb.Component()
        self._pb.starting_mode = r
        self._pb.n_mode = c.m
        self._serialize(c)
        return self._pb

    def _convert_bs_convention(self, convention):
        if convention == comp.BSConvention.H:
            return pb.BeamSplitter.H
        elif convention == comp.BSConvention.Ry:
            return pb.BeamSplitter.Ry
        return pb.BeamSplitter.Rx

    @dispatch(comp.BS)
    def _serialize(self, bs: comp.BS):
        pb_bs = pb.BeamSplitter()
        pb_bs.convention = self._convert_bs_convention(bs.convention)
        pb_bs.theta.CopyFrom(serialize_parameter(bs._theta))
        pb_bs.phi_tl.CopyFrom(serialize_parameter(bs._phi_tl))
        pb_bs.phi_bl.CopyFrom(serialize_parameter(bs._phi_bl))
        pb_bs.phi_tr.CopyFrom(serialize_parameter(bs._phi_tr))
        pb_bs.phi_br.CopyFrom(serialize_parameter(bs._phi_br))
        self._pb.beam_splitter.CopyFrom(pb_bs)

    @dispatch(comp.PS)
    def _serialize(self, ps: comp.PS):
        pb_ps = pb.PhaseShifter()
        pb_ps.phi.CopyFrom(serialize_parameter(ps._phi))
        if ps._max_error:
            pb_ps.max_error.CopyFrom(serialize_parameter(ps._max_error))
        self._pb.phase_shifter.CopyFrom(pb_ps)

    @dispatch(comp.PERM)
    def _serialize(self, p: comp.PERM):
        pb_perm = pb.Permutation()
        pb_perm.permutations.extend(p.perm_vector)
        self._pb.permutation.CopyFrom(pb_perm)

    @dispatch(comp.Unitary)
    def _serialize(self, unitary: comp.Unitary):
        pb_umat = serialize_matrix(unitary.U)
        pb_unitary = pb.Unitary()
        pb_unitary.mat.CopyFrom(pb_umat)
        if unitary.name != comp.Unitary.DEFAULT_NAME:
            pb_unitary.name = unitary.name
        pb_unitary.use_polarization = unitary.requires_polarization
        self._pb.unitary.CopyFrom(pb_unitary)

    @dispatch(comp.PBS)
    def _serialize(self, _):
        pb_pbs = pb.PolarizedBeamSplitter()
        self._pb.polarized_beam_splitter.CopyFrom(pb_pbs)

    @dispatch(comp.QWP)
    def _serialize(self, wp: comp.QWP):
        pb_wp = pb.WavePlate()
        pb_wp.xsi.CopyFrom(serialize_parameter(wp._xsi))
        self._pb.quarter_wave_plate.CopyFrom(pb_wp)

    @dispatch(comp.HWP)
    def _serialize(self, wp: comp.HWP):
        pb_wp = pb.WavePlate()
        pb_wp.xsi.CopyFrom(serialize_parameter(wp._xsi))
        self._pb.half_wave_plate.CopyFrom(pb_wp)

    @dispatch(comp.WP)
    def _serialize(self, wp: comp.WP):
        pb_wp = pb.WavePlate()
        pb_wp.delta.CopyFrom(serialize_parameter(wp._delta))
        pb_wp.xsi.CopyFrom(serialize_parameter(wp._xsi))
        self._pb.wave_plate.CopyFrom(pb_wp)

    @dispatch(nu.TD)
    def _serialize(self, td: nu.TD):
        pb_td = pb.TimeDelay()
        pb_td.dt.CopyFrom(serialize_parameter(td._dt))
        self._pb.time_delay.CopyFrom(pb_td)

    @dispatch(nu.LC)
    def _serialize(self, lc: nu.LC):
        pb_lc = pb.LossChannel()
        pb_lc.loss.CopyFrom(serialize_parameter(lc._loss))
        self._pb.loss_channel.CopyFrom(pb_lc)

    @dispatch(comp.PR)
    def _serialize(self, pr: comp.PR):
        pb_pr = pb.PolarizationRotator()
        pb_pr.delta.CopyFrom(serialize_parameter(pr._delta))
        self._pb.polarization_rotator.CopyFrom(pb_pr)

    @dispatch(comp.Barrier)
    def _serialize(self, barrier: comp.Barrier):
        pb_barrier = pb.Barrier()
        pb_barrier.visible = barrier.visible
        self._pb.barrier.CopyFrom(pb_barrier)

    @dispatch(FFConfigurator)
    def _serialize(self, ffconfigurator: FFConfigurator):
        pb_ffc = pb.FFConfigurator()
        pb_ffc.name = ffconfigurator.name
        pb_ffc.offset = ffconfigurator._offset
        pb_ffc.block_circuit_size = ffconfigurator._blocked_circuit_size

        pb_controlled = serialize_circuit(ffconfigurator._controlled)
        pb_ffc.controlled_circuit.CopyFrom(pb_controlled)

        pb_default_config = pb.VariableValues()
        for name, value in ffconfigurator._default_config.items():
            pb_default_config.mapping[name] = value
        pb_ffc.default_config.CopyFrom(pb_default_config)

        for state, mapping in ffconfigurator._configs.items():
            pb_vars = pb.VariableValues()
            for name, value in mapping.items():
                pb_vars.mapping[name] = value
            pb_ffc.configs[str(state)].CopyFrom(pb_vars)
        self._pb.ff_configurator.CopyFrom(pb_ffc)

    @dispatch(FFCircuitProvider)
    def _serialize(self, ffcp: FFCircuitProvider):
        from ._experiment_serialization import serialize_experiment
        pb_ffcp = pb.FFCircuitProvider()
        pb_ffcp.name = ffcp.name
        pb_ffcp.offset = ffcp._offset
        pb_ffcp.block_circuit_size = ffcp._blocked_circuit_size

        dc = ffcp.default_circuit
        if isinstance(dc, ACircuit):
            pb_ffcp.circuit.CopyFrom(serialize_circuit(dc))
        else:
            pb_ffcp.experiment.CopyFrom(serialize_experiment(dc))

        for state, circ in ffcp._map.items():
            pb_coe = pb.CircuitOrExperiment()
            if isinstance(circ, ACircuit):
                pb_coe.circuit.CopyFrom(serialize_circuit(circ))
            else:
                pb_coe.experiment.CopyFrom(serialize_experiment(circ))
            pb_ffcp.config_circ[str(state)].CopyFrom(pb_coe)
        self._pb.ff_circuit_provider.CopyFrom(pb_ffcp)

    @dispatch(Circuit)
    def _serialize(self, circuit: Circuit):
        pb_circ = serialize_circuit(circuit)
        self._pb.circuit.CopyFrom(pb_circ)


def serialize_circuit(circuit: ACircuit) -> pb.Circuit:
    if not isinstance(circuit, Circuit):
        circuit = Circuit(circuit.m).add(0, circuit)

    pb_circuit = pb.Circuit()
    if circuit.name != Circuit.DEFAULT_NAME:
        pb_circuit.name = circuit.name
    pb_circuit.n_mode = circuit.m
    comp_serializer = ComponentSerializer()
    for r, c in circuit._components:
        pb_circuit.components.extend([comp_serializer.serialize(r[0], c)])
    return pb_circuit


def serialize_component(component: AComponent) -> pb.Component:
    return ComponentSerializer().serialize(0, component)


def serialize_herald(herald: Herald) -> pb.Herald:
    pb_herald = pb.Herald()
    pb_herald.autogenerated_name = herald._autogenerated_name
    if herald.user_given_name is not None:
        pb_herald.name = herald.user_given_name
    pb_herald.value = herald._value
    return pb_herald


def serialize_port(port: Port) -> pb.Port:
    pb_port = pb.Port()
    pb_port.name = port.name
    pb_port.encoding = port.encoding.value
    return pb_port
