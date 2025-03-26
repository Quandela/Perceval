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

import pytest
import random
import sympy as sp
import numpy

from _test_utils import assert_circuits_eq, assert_experiment_equals
from perceval.components import ACircuit, Circuit, BSLayeredPPNR, Detector, BS, PS, TD, LC, Port, Herald, Experiment, \
    catalog, FFConfigurator, FFCircuitProvider
from perceval.utils import Matrix, E, P, BasicState, BSDistribution, BSCount, BSSamples, SVDistribution, StateVector, \
    NoiseModel, PostSelect, Encoding
from perceval.serialization import serialize, deserialize, serialize_binary, deserialize_circuit, deserialize_matrix
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


def _build_test_circuit():
    c1 = (Circuit(3) // comp.BS(theta=1.814) // comp.PS(phi=0.215, max_error=0.33) // comp.PERM([2, 0, 1])
          // (1, comp.PBS()) // comp.Unitary(Matrix.random_unitary(3)))
    c2 = Circuit(2) // comp.BS.H(theta=0.36, phi_tl=1.94, phi_br=5.8817, phi_bl=0.0179) // comp.PERM([1, 0])
    c1.add(1, c2, merge=False).add(0, comp.HWP(xsi=0.23)).add(1, comp.QWP(xsi=0.17)).add(2, comp.WP(0.4, 0.5))
    c1.add(0, comp.Barrier(2, visible=True))
    c1.add(0, comp.PR(delta=0.89))
    return c1


def test_circuit_serialization():
    c1 = _build_test_circuit()
    serialized_c1 = serialize(c1)
    deserialized_c1 = deserialize(serialized_c1)
    assert_circuits_eq(c1, deserialized_c1)


def test_non_unitary_serialization():
    t = TD(1)
    t_2 = deserialize(serialize(t))
    assert t is not t_2
    assert float(t_2._dt) == float(t._dt)

    l = LC(0.7)
    l_2 = deserialize(serialize(l))
    assert l is not l_2
    assert float(l_2._loss) == float(l._loss)


def test_circuit_serialization_backward_compat():
    serial_circuits = {
        # Perceval version (key) that generated the serialized representation of a given circuit (value)
        "0.7": ":PCVL:ACircuit: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",
        #0.8 : Did not change circuit serialization
        #0.9 : Did not change circuit serialization
        #0.10 : Did not change circuit serialization
        #0.11 : Updated parameter serialization
        "0.11": ":PCVL:zip:eJxVjs0KgkAURh/IlYhgAy5uV9GZrGYqB2cZEs1PYaWhvn2ukvl2Bw6Hj3CUFQE0n/ZrBpLDaI4ToLLtGYVTVX2qg7mM+dQxFPAfvTu2ErwXd6WM+q52AmQnUe/Zq4QNDjc7ZonBIoybaNup5vCgthfwpFG+dAsMc8l7HVySq9dFv7vzP8yeC2n6A8+rQV4=",
    }
    for perceval_version, serial_c in serial_circuits.items():
        try:
            deserialize(serial_c)
        except Exception as e:
            pytest.fail(f"Circuit serial representation generated with Perceval {perceval_version} failed: {e}")

def test_port_serialization():
    p = Port(Encoding.DUAL_RAIL, name = "test")
    p_2 = deserialize(serialize(p))
    assert p is not p_2
    assert p.name == p_2.name
    assert p.encoding == p_2.encoding

    h = Herald(2, name = "test")
    h_2 = deserialize(serialize(h))
    assert h is not h_2
    assert h.user_given_name == h_2.user_given_name
    assert h.expected == h_2.expected

    h = Herald(0)  # Same without user-given name
    h_2 = deserialize(serialize(h))
    assert h is not h_2
    assert h.user_given_name == h_2.user_given_name
    assert h._name == h_2._name
    assert h.expected == h_2.expected


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

    svd_empty = SVDistribution()
    deserialized_svd_empty = deserialize(serialize(svd_empty))
    assert deserialized_svd_empty == svd_empty


def test_bsdistribution_serialization():
    bsd = BSDistribution()
    bsd.add(BasicState([0, 1]), 0.4)
    bsd.add(BasicState([1, 0]), 0.4)
    bsd.add(BasicState([1, 1]), 0.2)
    deserialized_bsd = deserialize(serialize(bsd))
    assert bsd == deserialized_bsd

    bsd_empty = BSDistribution()
    deserialized_bsd_empty = deserialize(serialize(bsd_empty))
    assert bsd_empty == deserialized_bsd_empty


def test_bscount_serialization():
    bsc = BSCount()
    bsc.add(BasicState([0, 1]), 95811)
    bsc.add(BasicState([1, 0]), 56598)
    bsc.add(BasicState([1, 1]), 10558)
    deserialized_bsc = deserialize(serialize(bsc))
    assert bsc == deserialized_bsc

    bsc_empty = BSCount()
    deserialized_bsc_empty = deserialize(serialize(bsc_empty))
    assert deserialized_bsc_empty == bsc_empty


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

    empty_samples = BSSamples()
    deserialized_empty_samples = deserialize(serialize(empty_samples))
    assert empty_samples == deserialized_empty_samples


def test_sv_serialization():
    sv = (1+1j) * StateVector("|0,1>") + (1-1j) * StateVector("|1,0>")
    sv_serialized = serialize(sv)
    assert sv_serialized == ":PCVL:StateVector:(0.5,0.5)*|0,1>+(0.5,-0.5)*|1,0>" \
        or sv_serialized == ":PCVL:StateVector:(0.5,-0.5)*|1,0>+(0.5,0.5)*|0,1>"  # Order does not matter
    sv_deserialized = deserialize(sv_serialized)
    assert sv == sv_deserialized


def test_noise_model_serialization():
    empty_nm = NoiseModel()
    empty_nm_ser = serialize(empty_nm)
    empty_nm_deser = deserialize(empty_nm_ser)
    assert empty_nm == empty_nm_deser

    nm = NoiseModel(brightness=0.1, indistinguishability=0.2, g2=0.3, g2_distinguishable=True, transmittance=0.4,
                    phase_imprecision=0.5, phase_error=0.08)
    nm_ser = serialize(nm)
    nm_deser = deserialize(nm_ser)
    assert nm == nm_deser


@pytest.mark.parametrize("ps", (PostSelect(),
                                PostSelect("[0]==0 & [1,2]==1 & [3,4]==1 & [5]==0"),
                                PostSelect("[0]>0&[1]<2&[2]==1")))
def test_postselect_serialization(ps):
    ps_ser = serialize(ps)
    ps_deser = deserialize(ps_ser)
    assert ps == ps_deser


@pytest.mark.parametrize("detector", (BSLayeredPPNR(2, .6),
                                      Detector.pnr(),
                                      Detector.threshold(),
                                      Detector.ppnr(4, 2)))
def test_detector_serialization(detector):
    serialized = serialize(detector)
    deserialized = deserialize(serialized)
    assert detector.name == deserialized.name, "Wrong deserialized detector name"
    if isinstance(detector, BSLayeredPPNR):
        assert detector._layers == deserialized._layers and detector._r == deserialized._r, \
            "Wrong deserialized detector parameters"
    else:
        assert detector._wires == deserialized._wires and detector.max_detections == deserialized.max_detections, \
            "Wrong deserialized detector parameters"


def test_experiment_serialization():
    # Empty experiment
    e = Experiment()
    ser_e = serialize(e)
    e_2 = deserialize(ser_e)
    assert_experiment_equals(e, e_2)

    # Fully complete experiment
    e = Experiment(4, NoiseModel(.4, .5, .6), name="test")

    e.min_detected_photons_filter(2)

    # Includes components, heralds, postselection, ports
    e.add(0, catalog["postprocessed cnot"].build_processor())
    e.add(0, LC(.1))  # With non-unitary component
    e.with_input(BasicState([1, 0, 1, 0]))

    e.add(0, Detector.ppnr(24, 4))
    e.add(1, Detector.threshold())

    mzi = catalog['mzi phase last'].build_circuit()
    ffc = FFConfigurator(2, 0, mzi, {"phi_a": 0, "phi_b": 0}, "ffc name")
    ffc.add_configuration((0, 1), {"phi_a": 3, "phi_b": 0})
    ffc.add_configuration((1, 0), {"phi_a": 0, "phi_b": 3})
    e.add(0, ffc)

    ffcp = FFCircuitProvider(1, 0, Circuit(2) // BS.H(), "ffcp name")
    ffcp.add_configuration((0,), Circuit(2) // BS.Rx())
    subexp = Experiment(2)
    subexp.add(0, BS.Ry())
    ffcp.add_configuration((1,), subexp)

    e.add(1, ffcp)

    e.add(2, Detector.pnr())
    e.add(3, BSLayeredPPNR(2))

    ser_e = serialize(e)
    e_2 = deserialize(ser_e)
    assert_experiment_equals(e, e_2)


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


def test_binary_serialization():
    c_before = _build_test_circuit()
    bin_serialization = serialize_binary(c_before)
    assert isinstance(bin_serialization, bytes)
    c_after = deserialize_circuit(bin_serialization)
    assert_circuits_eq(c_before, c_after)
    with pytest.raises(TypeError):
        deserialize(bin_serialization)

    m_before = Matrix([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]])
    bin_serialization = serialize_binary(m_before)
    assert isinstance(bin_serialization, bytes)
    m_after = deserialize_matrix(bin_serialization)
    assert numpy.allclose(m_before, m_after)


def test_compress():
    zip_prefix = ":PCVL:zip:"

    c = _build_test_circuit()
    assert serialize(c).startswith(zip_prefix)  # Default value is to compress circuits
    assert serialize(c, compress=True).startswith(zip_prefix)
    assert serialize(c, compress=["ACircuit"]).startswith(zip_prefix)
    assert serialize(c, compress=["toto", "tutu", "ACircuit", "papa"]).startswith(zip_prefix)
    assert not serialize(c, compress=False).startswith(zip_prefix)
    assert not serialize(c, compress=[]).startswith(zip_prefix)
    assert not serialize(c, compress=["toto", "tutu", "papa"]).startswith(zip_prefix)
    with pytest.raises(NotImplementedError):
        serialize(c, compress=12).startswith(zip_prefix)

    d = {
        "input_state": BasicState([0, 1, 0, 1]),
        "circuit": Circuit(4) // (0, comp.BS()) // (2, comp.BS()),
        "integer": 43
    }
    d_compressed = serialize(d, compress=True)
    assert d_compressed["input_state"].startswith(zip_prefix)
    assert d_compressed["circuit"].startswith(zip_prefix)
    assert d_compressed["integer"] == 43  # JSon-compatible integral types are neither serialized nor compressed

    d_not_compressed = serialize(d, compress=False)
    assert not d_not_compressed["input_state"].startswith(zip_prefix)
    assert not d_not_compressed["circuit"].startswith(zip_prefix)
    assert d_not_compressed["integer"] == 43  # JSon-compatible integral types are neither serialized nor compressed

    # Compressing a circuit serialization gives a smaller representation
    assert len(d_compressed["circuit"]) < len(d_not_compressed['circuit'])

    d_only_circuit = serialize(d, compress=["ACircuit"])  # Compress only ACircuit objects
    assert not d_only_circuit["input_state"].startswith(zip_prefix)
    assert d_only_circuit["circuit"].startswith(zip_prefix)

    d_only_basicstate = serialize(d, compress=["BasicState"])  # Compress only BasicState objects
    assert d_only_basicstate["input_state"].startswith(zip_prefix)
    assert not d_only_basicstate["circuit"].startswith(zip_prefix)


def test_circuit_with_expression_serialization():
    p_a = P("A")
    p_b = P("B")
    sum_ab = E("A + B", {p_a, p_b})

    c = Circuit(2)
    c.add(0, PS(phi=p_a))
    c.add(0, BS(theta=sum_ab))
    c.add(0, PS(phi=p_a))

    c_ser = serialize(c)
    c_deser = deserialize(c_ser)

    assert "A" in c_deser.params and "B" in c_deser.params

    a_value = 0.2
    b_value = 0.8
    c_deser.param("A").set_value(a_value)
    c_deser.param("B").set_value(b_value)

    assert float(c_deser._components[1][1].param('theta')) == pytest.approx(0.2 + 0.8)

    p_a.set_value(a_value)
    p_b.set_value(b_value)
    assert numpy.allclose(c_deser.compute_unitary(), c.compute_unitary())
