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
import numpy as np

from perceval.utils import Matrix
from perceval.serialization._parameter_serialization import deserialize_parameter


def serialize_matrix(m: Matrix) -> pb.Matrix:
    pb_mat = pb.Matrix()
    pb_mat.rows = m.shape[0]
    pb_mat.cols = m.shape[1]
    values = []
    if m.is_symbolic():
        pb_symbolic = pb.MatrixSymbolic()
        for x in m.vec():
            pb_param = pb.Parameter()
            pb_param.expression = str(x)
            values.append(pb_param)
        pb_symbolic.data.extend(values)
        pb_mat.symbolic.CopyFrom(pb_symbolic)
    else:
        pb_numeric = pb.MatrixDouble()
        values = []
        for x in np.nditer(m):
            pb_complex = pb.ComplexDouble()
            pb_complex.real_value = float(x.real)
            pb_complex.imaginary_value = float(x.imag)
            values.append(pb_complex)
        pb_numeric.data.extend(values)
        pb_mat.numeric.CopyFrom(pb_numeric)
    return pb_mat


def _deserialize_numeric(pb_mat):
    assert len(pb_mat.numeric.data) == pb_mat.rows * pb_mat.cols, "Unexpected number of elements in serialized matrix"
    ncols = pb_mat.cols
    array = []
    row = []
    for cplx in pb_mat.numeric.data:
        row.append(complex(cplx.real_value, cplx.imaginary_value))
        if len(row) == ncols:
            array.append(row)
            row = []
    return array


def _deserialize_symbolic(pb_mat):
    assert len(pb_mat.symbolic.data) == pb_mat.rows * pb_mat.cols, "Unexpected number of elements in serialized matrix"
    ncols = pb_mat.cols
    array = []
    row = []
    for param in pb_mat.symbolic.data:
        row.append(deserialize_parameter(param).spv)
        if len(row) == ncols:
            array.append(row)
            row = []
    return array


def deserialize_pb_matrix(pb_mat: pb.Matrix) -> Matrix:
    if pb_mat.HasField('numeric'):
        mat_data = _deserialize_numeric(pb_mat)
        return Matrix(mat_data)
    else:
        mat_data = _deserialize_symbolic(pb_mat)
        return Matrix(mat_data, use_symbolic=True)
