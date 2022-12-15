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

import random
import sympy as sp
import numpy
from perceval import Matrix, P, ACircuit, Circuit
from perceval.utils.statevector import BasicState, BSDistribution, BSCount, BSSamples, SVDistribution, StateVector
from perceval.serialization import serialize, deserialize
from perceval.serialization._parameter_serialization import serialize_parameter, deserialize_parameter
import perceval.components.unitary_components as comp
import json


def test_numeric_matrix_serialization():
    input_mat = Matrix.random_unitary(10)
    serialized_mat = serialize(input_mat)
    deserialized_mat = deserialize(serialized_mat)
    assert (input_mat == deserialized_mat).all()

    input_mat = Matrix([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]])
    serialized_mat = serialize(input_mat)
    deserialized_mat = deserialize(serialized_mat)
    assert (input_mat == deserialized_mat).all()


def test_symbolic_matrix_serialization():
    theta = P('theta')
    bs = comp.BS(theta=theta)
    input_mat = bs.U
    serialized_mat = serialize(input_mat)
    deserialized_mat = deserialize(serialized_mat)

    # Now, assign any value to theta:
    theta_value = random.random()
    theta.set_value(theta_value)
    input_mat_num = bs.compute_unitary()
    convert_to_numpy = sp.lambdify((), deserialized_mat.subs({'theta': theta_value}), modules=numpy)
    deserialized_mat_num = Matrix(convert_to_numpy())
    assert numpy.allclose(input_mat_num, deserialized_mat_num)


def test_symbol_serialization():
    theta = P('theta')
    theta_deserialized = deserialize_parameter(serialize_parameter(theta))
    assert theta_deserialized.is_symbolic()
    assert theta._symbol == theta_deserialized._symbol


def _check_circuits_eq(c_a, c_b):
    assert c_a.ncomponents() == c_b.ncomponents()
    for nc in range(len(c_a._components)):
        input_idx, input_comp = c_a._components[nc]
        output_idx, output_comp = c_b._components[nc]
        assert isinstance(input_comp, type(output_comp))
        assert list(input_idx) == list(output_idx)
        assert (input_comp.compute_unitary() == output_comp.compute_unitary()).all()


def _build_test_circuit():
    c1 = Circuit(3) // comp.BS(theta=1.814) // comp.PS(phi=0.215) // comp.PERM([2, 0, 1]) // (1, comp.PBS()) \
         // comp.Unitary(Matrix.random_unitary(3))
    c2 = Circuit(2) // comp.BS.H(theta=0.36, phi_tl=1.94, phi_br=5.8817, phi_bl=0.0179) // comp.PERM([1, 0])
    c1.add(1, c2, merge=False).add(0, comp.HWP(xsi=0.23)).add(1, comp.QWP(xsi=0.17)).add(2, comp.WP(0.4, 0.5))
    c1.add(0, comp.PR(delta=0.89))
    return c1


def test_circuit_serialization():
    c1 = _build_test_circuit()
    serialized_c1 = serialize(c1)
    deserialized_c1 = deserialize(serialized_c1)
    _check_circuits_eq(c1, deserialized_c1)


def test_basicstate_serialization():
    states = [
        BasicState("|0,1>"),
        BasicState([0, 1, 0, 0, 1, 0]),
        BasicState("|{P:H}{P:V},0>")
    ]
    for s in states:
        serialized = serialize(s)
        deserialized = deserialize(serialized)
        assert s == deserialized


def test_svdistribution_serialization():
    svd = SVDistribution()
    svd[StateVector("|0,1>")] = 0.2
    svd[BasicState("|1,0>")] = 0.3
    svd[BasicState("|1,1>")] = 0.5
    svd2 = deserialize(serialize(svd))
    assert svd == svd2


def test_bsdistribution_serialization():
    bsd = BSDistribution()
    bsd.add(BasicState([0, 1]), 0.4)
    bsd.add(BasicState([1, 0]), 0.4)
    bsd.add(BasicState([1, 1]), 0.2)
    deserialized_bsd = deserialize(serialize(bsd))
    assert bsd == deserialized_bsd


def test_bscount_serialization():
    bsc = BSCount()
    bsc.add(BasicState([0, 1]), 95811)
    bsc.add(BasicState([1, 0]), 56598)
    bsc.add(BasicState([1, 1]), 10558)
    deserialized_bsc = deserialize(serialize(bsc))
    assert bsc == deserialized_bsc


def test_bssamples_serialization():
    samples = BSSamples()
    for j in range(50):
        for i in range(11):
            samples.append(BasicState([0, 1, 0]))
        for i in range(13):
            samples.append(BasicState([1, 0, 0]))
        for i in range(17):
            samples.append(BasicState([0, 0, 1]))
    deserialized_samples = deserialize(serialize(samples))
    assert deserialized_samples == samples


def test_sv_serialization():
    sv = (1+1j) * StateVector("|0,1>") + (1-1j) * StateVector("|1,0>")
    sv_serialized = serialize(sv)
    assert sv_serialized == ":PCVL:StateVector:(0.5,0.5)*|0,1>+(0.5,-0.5)*|1,0>"
    sv_deserialized = deserialize(sv_serialized)
    assert sv == sv_deserialized


def test_json():
    svd = SVDistribution()
    svd.add(BasicState("|1,0>"), 0.5)
    svd.add(BasicState("|0,1>"), 0.5)
    encoded = serialize({"a": BasicState("|1,0>"),
                                   "b": Circuit(2) // comp.BS(),
                                   "c": Matrix.random_unitary(3),
                                   "d": svd
                                  })
    s = json.dumps(encoded)
    d = deserialize(json.loads(s))
    assert isinstance(d["a"], BasicState)
    assert isinstance(d["b"], ACircuit)
    assert isinstance(d["c"], Matrix)
    assert isinstance(d["d"], SVDistribution)
