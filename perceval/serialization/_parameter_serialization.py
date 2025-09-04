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
import sympy as sp
from perceval.utils import Parameter, Expression
from perceval.serialization import _schema_circuit_pb2 as pb


def serialize_parameter(param: Parameter | float):
    """
    Serialize a numerical value, a Parameter or an Expression as a protobuf "Parameter" message

    :param param: Different cases
        * A numerical value is serialized as an anonymous Parameter having the given value
        * A Parameter is serialized as is, with at least a symbol name, or a numerical value
        * An Expression serializes all its sub-Parameters internally + its sympy parsable expression
    """
    pb_param = pb.Parameter()
    if isinstance(param, float):
        pb_param.real_value = param
        return pb_param

    if param._symbol is not None:
        pb_param.name = str(param._symbol)

    if param.defined:
        pb_param.real_value = float(param)

    elif param._is_expression:
        serial_param_list = []
        for internal_param in param.parameters:
            serial_param_list.append(serialize_parameter(internal_param))
        pb_param.expr_parameters.extend(serial_param_list)
        pb_param.expression = param.name

    else:  # If param has no value and is not an expression, it is defined only by a symbol
        pb_param.symbol = str(param._symbol)
    return pb_param


def deserialize_parameter(serial_param: pb.Parameter, known_params = None):
    """
    Deserialize a protobuf "Parameter" message to a Parameter, an Expression or a numerical value.

    :param serial_param: Protobuf Parameter message
    :param known_params: A dictionary of already known parameters, used to reuse the same Parameter
                         instance, when the same parameter name is met more than once.
    """
    if known_params is None:
        known_params = {}

    t = serial_param.WhichOneof('type')
    if t == 'real_value':
        if serial_param.name:
            if serial_param.name in known_params:
                if float(known_params[serial_param.name]) != serial_param.real_value:
                    raise ValueError(f"Parameter {serial_param.name} has multiple values ({float(known_params[serial_param.name])} and {serial_param.real_value})")
                return known_params[serial_param.name]
            p = Parameter(serial_param.name)
            p.set_value(serial_param.real_value)
            known_params[serial_param.name] = p
            return p

        return serial_param.real_value

    elif t == 'symbol':
        if serial_param.symbol in known_params:
            return known_params[serial_param.symbol]
        p = Parameter(serial_param.symbol)
        known_params[serial_param.name] = p
        return p

    elif t == 'expression':
        if serial_param.expr_parameters:
            internal_params = set()
            for internal_serial_param in serial_param.expr_parameters:
                internal_params.add(deserialize_parameter(internal_serial_param, known_params))
            return Expression(serial_param.name, internal_params)
        return sp.S(serial_param.expression)
