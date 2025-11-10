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

from perceval.serialization import _schema_circuit_pb2 as pb
from perceval.serialization._parameter_serialization import deserialize_parameter
from perceval.serialization._matrix_serialization import deserialize_pb_matrix
import perceval.components.unitary_components as comp
import perceval.components.non_unitary_components as nu
from perceval.components.feed_forward_configurator import FFConfigurator, FFCircuitProvider
from perceval.utils import BasicState


def deserialize_ps(serial_ps: pb.PhaseShifter, known_params: dict = None) -> comp.PS:
    max_error = deserialize_parameter(serial_ps.max_error, known_params)
    if max_error is not None:
        return comp.PS(deserialize_parameter(serial_ps.phi, known_params), max_error)
    return comp.PS(deserialize_parameter(serial_ps.phi, known_params))


def _convert_bs_convention(ser_convention):
    if ser_convention == pb.BeamSplitter.Ry:
        return comp.BSConvention.Ry
    elif ser_convention == pb.BeamSplitter.H:
        return comp.BSConvention.H
    return comp.BSConvention.Rx


def deserialize_bs(serial_bs: pb.BeamSplitter, known_params: dict = None) -> comp.BS:
    conv = _convert_bs_convention(serial_bs.convention)
    return comp.BS(theta=deserialize_parameter(serial_bs.theta, known_params),
                   phi_tl=deserialize_parameter(serial_bs.phi_tl, known_params),
                   phi_bl=deserialize_parameter(serial_bs.phi_bl, known_params),
                   phi_tr=deserialize_parameter(serial_bs.phi_tr, known_params),
                   phi_br=deserialize_parameter(serial_bs.phi_br, known_params),
                   convention=conv)


def deserialize_perm(serial_perm, _) -> comp.PERM:
    return comp.PERM([x for x in serial_perm.permutations])


def deserialize_unitary(serial_unitary, _) -> comp.Unitary:
    m = deserialize_pb_matrix(serial_unitary.mat)
    return comp.Unitary(U=m)


def deserialize_wp(serial_wp, known_params: dict = None) -> comp.WP:
    return comp.WP(deserialize_parameter(serial_wp.delta, known_params),
                   deserialize_parameter(serial_wp.xsi, known_params))


def deserialize_qwp(serial_qwp, known_params: dict = None) -> comp.QWP:
    return comp.QWP(deserialize_parameter(serial_qwp.xsi, known_params))


def deserialize_hwp(serial_hwp, known_params: dict = None) -> comp.HWP:
    return comp.HWP(deserialize_parameter(serial_hwp.xsi, known_params))


def deserialize_dt(serial_dt, known_params: dict = None) -> nu.TD:
    return nu.TD(deserialize_parameter(serial_dt.dt, known_params))


def deserialize_lc(serial_lc, known_params: dict = None) -> nu.LC:
    return nu.LC(deserialize_parameter(serial_lc.loss, known_params))


def deserialize_pr(serial_pr, known_params: dict = None) -> comp.PR:
    return comp.PR(deserialize_parameter(serial_pr.delta, known_params))


def deserialize_pbs(_, __) -> comp.PBS:
    return comp.PBS()


def deserialize_barrier(m: int, serial_barrier, _) -> comp.Barrier:
    return comp.Barrier(m, serial_barrier.visible)


def deserialize_ff_configurator(m: int, serial_ffc, known_params: dict = None) -> FFConfigurator:
    from .deserialize import deserialize_circuit
    default_config = dict(serial_ffc.default_config.mapping)
    ffc = FFConfigurator(m, serial_ffc.offset, deserialize_circuit(serial_ffc.controlled_circuit, known_params),
                         default_config, serial_ffc.name or None)
    for state_str, config in serial_ffc.configs.items():
        config_dict = dict(config.mapping)
        ffc.add_configuration(BasicState(state_str), config_dict)
    if serial_ffc.block_circuit_size:
        ffc.block_circuit_size()

    return ffc


def deserialize_ff_circuit_provider(m: int, serial_ffcp, known_params: dict = None) -> FFCircuitProvider:
    from .deserialize import deserialize_circuit, deserialize_experiment
    if serial_ffcp.WhichOneof('default_circuit') == "circuit":
        default_circ = deserialize_circuit(serial_ffcp.circuit, known_params)
    else:
        default_circ = deserialize_experiment(serial_ffcp.experiment, known_params)
    ffcp = FFCircuitProvider(m, serial_ffcp.offset, default_circ, serial_ffcp.name or None)
    for state_str, serial_circ in serial_ffcp.config_circ.items():
        if serial_circ.WhichOneof('type') == "circuit":
            circ = deserialize_circuit(serial_circ.circuit, known_params)
        else:
            circ = deserialize_experiment(serial_circ.experiment, known_params)
        ffcp.add_configuration(BasicState(state_str), circ)
    if serial_ffcp.block_circuit_size:
        ffcp.block_circuit_size()
    return ffcp


def deserialize_compiled_circuit(serial_cc, known_params: dict = None) -> FFCircuitProvider:
    from perceval.serialization.deserialize import CompiledCircuitBuilder
    builder = CompiledCircuitBuilder(serial_cc, known_params)
    return builder.resolve()
