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
import perceval.lib.symb as symb
import perceval.lib.phys as phys


def deserialize_ps(serial_ps: pb.PhaseShifter, ns: int):
    return_type = phys.PS if ns == pb.Component.PHYS else symb.PS
    return return_type(deserialize_parameter(serial_ps.phi))


def deserialize_phys_bs(serial_bs: pb.BeamSplitterComplex) -> phys.BS:
    args = {}
    if serial_bs.HasField('R'):
        args['R'] = deserialize_parameter(serial_bs.R)
    if serial_bs.HasField('theta'):
        args['theta'] = deserialize_parameter(serial_bs.theta)
    args['phi_a'] = deserialize_parameter(serial_bs.phi_a)
    args['phi_b'] = deserialize_parameter(serial_bs.phi_b)
    args['phi_d'] = deserialize_parameter(serial_bs.phi_d)
    return phys.BS(**args)


def deserialize_perm(serial_perm, ns: int):
    return_type = phys.PERM if ns == pb.Component.PHYS else symb.PERM
    return return_type([x for x in serial_perm.permutations])


def deserialize_unitary(serial_unitary, ns: int):
    return_type = phys.Unitary if ns == pb.Component.PHYS else symb.Unitary
    m = deserialize_pb_matrix(serial_unitary.mat)
    return return_type(U=m)


def deserialize_symb_bs(serial_bs: pb.BeamSplitter) -> symb.BS:
    args = {}
    if serial_bs.HasField('R'):
        args['R'] = deserialize_parameter(serial_bs.R)
    if serial_bs.HasField('theta'):
        args['theta'] = deserialize_parameter(serial_bs.theta)
    return symb.BS(**args)


def deserialize_wp(serial_wp, ns):
    return_type = phys.WP if ns == pb.Component.PHYS else symb.WP
    return return_type(deserialize_parameter(serial_wp.delta), deserialize_parameter(serial_wp.xsi))


def deserialize_qwp(serial_qwp, ns):
    return_type = phys.QWP if ns == pb.Component.PHYS else symb.QWP
    return return_type(deserialize_parameter(serial_qwp.xsi))


def deserialize_hwp(serial_hwp, ns):
    return_type = phys.HWP if ns == pb.Component.PHYS else symb.HWP
    return return_type(deserialize_parameter(serial_hwp.xsi))


def deserialize_dt(serial_dt, ns):
    return_type = phys.TD if ns == pb.Component.PHYS else symb.TD
    return return_type(deserialize_parameter(serial_dt.dt))


def deserialize_pr(serial_pr, ns):
    return_type = phys.PR if ns == pb.Component.PHYS else symb.PR
    return return_type(deserialize_parameter(serial_pr.delta))


def deserialize_pbs(_, ns):
    return phys.PBS() if ns == pb.Component.PHYS else symb.PBS()
