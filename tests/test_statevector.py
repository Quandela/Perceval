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

from collections import Counter

import pytest
import perceval as pcvl
import perceval.components.unitary_components as comp
from perceval.rendering.pdisplay import pdisplay_state_distrib

import sympy as sp

from test_circuit import strip_line_12


def test_state():
    st = pcvl.BasicState([1, 0])
    assert str(st) == "|1,0>"
    assert st.n == 1
    assert st.has_annotations is False


def test_str_state_vector():
    sv = (1+1j) * pcvl.StateVector("|0,1>") + (1-1j) * pcvl.StateVector("|1,0>")
    assert str(sv) == "(1/2+I/2)*|0,1>+(1/2-I/2)*|1,0>"


def test_tensor_product_0():
    st1 = pcvl.BasicState([0, 1])
    st2 = pcvl.BasicState([1])
    assert str(st1*st2) == "|0,1,1>"


def test_tensor_product_1():
    st1 = pcvl.BasicState([1, 2])
    st2 = pcvl.BasicState([3, 4])
    assert str(st1*st2) == "|1,2,3,4>"


def test_tensor_product_2():
    st1 = pcvl.BasicState("|1,{P:V}1>")
    assert str(st1) == "|1,{P:V}1>"
    st2 = pcvl.BasicState("|3,{P:H}3>")
    assert str(st2) == "|3,{P:H}3>"
    assert str(st1*st2) == "|1,{P:V}1,3,{P:H}3>"


def test_tensor_svdistribution_1():
    sv1 = pcvl.SVDistribution()
    sv1.add(pcvl.StateVector([0]), 0.25)
    sv1.add(pcvl.StateVector([1]), 0.75)
    sv2 = pcvl.SVDistribution()
    sv2.add(pcvl.StateVector([0]), 0.8)
    sv2.add(pcvl.StateVector([1]), 0.2)
    sv = sv1*sv2
    assert len(sv) == 4
    assert pytest.approx(sv[pcvl.StateVector([0, 0])]) == 1/5
    assert pytest.approx(sv[pcvl.StateVector([0, 1])]) == 1/20
    assert pytest.approx(sv[pcvl.StateVector([1, 0])]) == 3/5
    assert pytest.approx(sv[pcvl.StateVector([1, 1])]) == 3/20


def test_state_annots():
    st = pcvl.BasicState('|0,1,{P:V}1>')
    assert st.n == 3
    assert st.m == 3
    assert st.has_annotations
    assert str(st) == '|0,1,{P:V}1>'
    assert [str(a) for a in st.get_mode_annotations(1)] == [""]
    assert [str(a) for a in st.get_mode_annotations(2)] == ["P:V", ""]
    # assert st.get_photon_annotations(2) == {'P': 'V'}
    # assert st.get_photon_annotations(3) == {}
    # st.set_photon_annotations(3, {"P": "H"})
    # assert str(st) == '|0,1,{P:H}{P:V}>'
    # st.set_photon_annotations(3, {"P": "V"})
    # assert str(st) == '|0,1,2{P:V}>'
    assert st.get_mode_annotations(1) == [pcvl.Annotation()]
    assert st.get_mode_annotations(2) == [pcvl.Annotation("P:V"), pcvl.Annotation()]


def test_state_identical_annots():
    st1 = pcvl.BasicState("|0,1,{P:V}1>")
    st2 = pcvl.BasicState("|0,1,{P:V}1>")
    assert str(st1) == str(st2)


def test_state_invalid_superposition():
    st1 = pcvl.StateVector("|0,1>")
    st2 = pcvl.StateVector("|1,0,0>")
    with pytest.raises(AssertionError):
        st1+st2


def test_state_superposition():
    st1 = pcvl.StateVector("|0,1>")
    st2 = pcvl.StateVector("|1,0>")
    assert str(2*st1) == str(st1)
    assert str(st1+1j*st2) == 'sqrt(2)/2*|0,1>+sqrt(2)*I/2*|1,0>'
    assert (str(pcvl.StateVector("|0,1>")+sp.S("ε")*pcvl.StateVector("|1,0>")) ==
            '(Abs(ε)**2 + 1)**(-0.5)*|0,1>+ε/(Abs(ε)**2 + 1)**0.5*|1,0>')


def test_state_superposition_sub():
    assert (str(pcvl.StateVector("|0,1>")-sp.S("ε")*pcvl.StateVector("|1,0>")) ==
            '(Abs(ε)**2 + 1)**(-0.5)*|0,1>-ε/(Abs(ε)**2 + 1)**0.5*|1,0>')


def test_state_superposition_bs():
    assert (str(pcvl.BasicState("|0,1>")-pcvl.BasicState("|1,0>")) ==
            'sqrt(2)/2*|0,1>-sqrt(2)/2*|1,0>')


def test_init_state_vector():
    st = pcvl.StateVector()
    st[pcvl.BasicState("|1,0>")] = 1
    assert str(st) == "|1,0>"


def test_svdistribution():
    st1 = pcvl.StateVector("|0,1>")
    st2 = pcvl.StateVector("|1,0>")
    svd = pcvl.SVDistribution()
    svd.add(st1, 0.5)
    svd[st2] = 0.5
    assert strip_line_12(pdisplay_state_distrib(svd)) == strip_line_12("""
            +-------+-------------+
            | state | probability |
            +-------+-------------+
            | |0,1> |     1/2     |
            | |1,0> |     1/2     |
            +-------+-------------+""")


def test_sv_separation_0():
    st1 = pcvl.BasicState("|0,0>")
    assert st1.separate_state() == [pcvl.BasicState("|0,0>")]


def test_sv_separation_1():
    st1 = pcvl.BasicState("|0,1>")
    assert st1.separate_state() == [pcvl.BasicState("|0,1>")]
    st2 = pcvl.BasicState("|2,1>")
    assert st2.separate_state() == [pcvl.BasicState("|2,1>")]


def test_sv_separation_2():
    st1 = pcvl.BasicState("|0,{_:1}>")
    assert st1.separate_state() == [pcvl.BasicState("|0,1>")]
    st2 = pcvl.BasicState("|{_:1},{P:V}>")
    assert st2.separate_state() == [pcvl.BasicState("|1,1>")]
    st3 = pcvl.BasicState("|{_:1},{_:2}>")
    assert st3.separate_state() == [pcvl.BasicState("|1,0>"), pcvl.BasicState("|0,1>")]


def test_sv_separation_3():
    st1 = pcvl.BasicState("|{_:1},{_:0}>")
    assert st1.separate_state() == [pcvl.BasicState("|1,0>"), pcvl.BasicState("|0,1>")]
    st2 = pcvl.BasicState("|{_:0},{_:0},{_:3}>")
    assert st2.separate_state() == [pcvl.BasicState("|1,1,0>"), pcvl.BasicState("|0,0,1>")]


def test_sv_split():
    st1 = pcvl.BasicState("|1,1,1>")
    partition = st1.partition([2, 1])
    expected = ["|1,1,0> |0,0,1>", "|1,0,1> |0,1,0>", "|0,1,1> |1,0,0>"]
    result = []
    for a_subset in partition:
        result.append(" ".join([str(state) for state in a_subset]))
    for r, e in zip(sorted(result), sorted(expected)):
        assert r == e


def test_sv_parse_annot():
    invalid_str = ["|{_:0}", "|0{_:0}>", "|1{_:2>", "{P:(0.3,0)>",
                   "|{;}>", "|{P:(1,2,3)}>", "|{P:(1,a)}>", "|{a:0,a:1}>"]
    for s in invalid_str:
        with pytest.raises(ValueError):
            pcvl.BasicState(s)
    st1 = pcvl.BasicState("|{_:0}{_:1}>")
    st1.clear_annotations()
    assert str(st1) == "|2>"
    st1 = pcvl.BasicState("|{_:0}{_:1},0,1>")
    st1.clear_annotations()
    assert str(st1) == "|2,0,1>"


def test_sv_parse_symb_annot():
    st1 = pcvl.BasicState("|{P:1.5707963268}>")
    assert str(st1) == "|{P:D}>"


def test_sv_parse_tuple_annot():
    st1 = pcvl.BasicState("|{P:(0.30,0)}>")
    assert str(st1) == "|{P:0.3}>"
    #st1 = pcvl.BasicState("|{P:(pi/2,0.3)}>")
    #assert str(st1) == "|{P:(pi/2,0.3)}>"


def test_sv_sample():
    source = pcvl.Source(brightness=1, purity=0.9, indistinguishability=0.9)
    qpu = pcvl.Processor("Naive", comp.BS(), source)
    qpu.with_input(pcvl.BasicState([1, 0]))
    sample = qpu.source_distribution.sample(1)
    assert isinstance(sample, pcvl.StateVector)
    sample = qpu.source_distribution.sample(2)
    assert len(sample) == 2
    assert isinstance(sample[0], pcvl.StateVector)
    assert isinstance(sample[1], pcvl.StateVector)


def test_statevector_sample():
    sv = pcvl.StateVector("|0,1>")+pcvl.StateVector("|1,0>")
    counter = Counter()
    for s in range(20):
        counter[sv.sample()] += 1
    states = [str(s) for s in counter]
    assert len(states) == 2 and "|1,0>" in states and "|0,1>" in states


def test_statevector_samples():
    sv = pcvl.StateVector("|0,1>") + pcvl.StateVector("|1,0>")
    counter = Counter()
    for s in sv.samples(20):
        counter[s] += 1
    states = [str(s) for s in counter]
    assert len(states) == 2 and "|1,0>" in states and "|0,1>" in states


def test_statevector_measure_1():
    sv = pcvl.StateVector("|0,1>")+pcvl.StateVector("|1,0>")
    map_measure_sv = sv.measure(0)
    assert len(map_measure_sv) == 2 and\
           pcvl.BasicState("|0>") in map_measure_sv and\
           pcvl.BasicState("|1>") in map_measure_sv
    assert pytest.approx(0.5) == map_measure_sv[pcvl.BasicState("|0>")][0]
    assert str(map_measure_sv[pcvl.BasicState("|0>")][1]) == "|1>"
    assert pytest.approx(0.5) == map_measure_sv[pcvl.BasicState("|1>")][0]
    assert str(map_measure_sv[pcvl.BasicState("|1>")][1]) == "|0>"


def test_statevector_measure_1():
    sv = pcvl.StateVector("|0,1>")+pcvl.StateVector("|1,0>")
    map_measure_sv = sv.measure([0, 1])
    assert len(map_measure_sv) == 2 and\
           pcvl.BasicState("|0,1>") in map_measure_sv and\
           pcvl.BasicState("|1,0>") in map_measure_sv
    assert pytest.approx(0.5) == map_measure_sv[pcvl.BasicState("|0,1>")][0]
    assert str(map_measure_sv[pcvl.BasicState("|0,1>")][1]) == "|>"
    assert pytest.approx(0.5) == map_measure_sv[pcvl.BasicState("|1,0>")][0]
    assert str(map_measure_sv[pcvl.BasicState("|1,0>")][1]) == "|>"


def test_statevector_measure_2():
    sv = pcvl.StateVector("|0,1,1>")+pcvl.StateVector("|1,1,0>")
    map_measure_sv = sv.measure(1)
    assert len(map_measure_sv) == 1 and\
           pcvl.BasicState("|1>") in map_measure_sv
    assert pytest.approx(1) == map_measure_sv[pcvl.BasicState("|1>")][0]
    assert str(map_measure_sv[pcvl.BasicState("|1>")][1]) == "sqrt(2)/2*|0,1>+sqrt(2)/2*|1,0>"
