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

from perceval.serialization import _schema_circuit_pb2 as pb
from perceval.serialization._parameter_serialization import deserialize_parameter
from perceval.serialization._matrix_serialization import deserialize_pb_matrix
import perceval.components.unitary_components as comp
import perceval.components.non_unitary_components as nu


def deserialize_ps(serial_ps: pb.PhaseShifter) -> comp.PS:
    return comp.PS(deserialize_parameter(serial_ps.phi))


def _convert_bs_convention(ser_convention):
    if ser_convention == pb.BeamSplitter.Ry:
        return comp.BSConvention.Ry
    elif ser_convention == pb.BeamSplitter.H:
        return comp.BSConvention.H
    return comp.BSConvention.Rx


def deserialize_bs(serial_bs: pb.BeamSplitter) -> comp.BS:
    conv = _convert_bs_convention(serial_bs.convention)
    return comp.BS(theta=deserialize_parameter(serial_bs.theta),
                   phi_tl=deserialize_parameter(serial_bs.phi_tl),
                   phi_bl=deserialize_parameter(serial_bs.phi_bl),
                   phi_tr=deserialize_parameter(serial_bs.phi_tr),
                   phi_br=deserialize_parameter(serial_bs.phi_br),
                   convention=conv)


def deserialize_perm(serial_perm) -> comp.PERM:
    return comp.PERM([x for x in serial_perm.permutations])


def deserialize_unitary(serial_unitary) -> comp.Unitary:
    m = deserialize_pb_matrix(serial_unitary.mat)
    return comp.Unitary(U=m)


def deserialize_wp(serial_wp) -> comp.WP:
    return comp.WP(deserialize_parameter(serial_wp.delta), deserialize_parameter(serial_wp.xsi))


def deserialize_qwp(serial_qwp) -> comp.QWP:
    return comp.QWP(deserialize_parameter(serial_qwp.xsi))


def deserialize_hwp(serial_hwp) -> comp.HWP:
    return comp.HWP(deserialize_parameter(serial_hwp.xsi))


def deserialize_dt(serial_dt) -> nu.TD:
    return comp.TD(deserialize_parameter(serial_dt.dt))


def deserialize_pr(serial_pr) -> comp.PR:
    return comp.PR(deserialize_parameter(serial_pr.delta))


def deserialize_pbs(_) -> comp.PBS:
    return comp.PBS()
