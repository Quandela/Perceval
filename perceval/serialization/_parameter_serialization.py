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
from typing import Union

from perceval.utils import Parameter, Expression
from perceval.serialization import _schema_circuit_pb2 as pb


def serialize_parameter(param: Union[Parameter, float]):
    pb_param = pb.Parameter()
    if isinstance(param, float):
        pb_param.real_value = param
    else:
        if param.defined:
            pb_param.real_value = float(param)
        elif param._is_expression:
            pb_param.expression = str(param.spv)
        else:  # If param has no value and is not an expression, it is defined only by a symbol
            pb_param.symbol = str(param._symbol)
    return pb_param


def deserialize_parameter(serial_param: pb.Parameter):
    t = serial_param.WhichOneof('type')
    if t == 'real_value':
        return serial_param.real_value
    elif t == 'symbol':
        return Parameter(serial_param.symbol)
    elif t == 'expression':
        return Expression(serial_param.expression)
