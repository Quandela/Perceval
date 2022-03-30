import pytest
import perceval as pcvl
import perceval.lib.symb as symb

import sympy as sp

from test_circuit import strip_line_12


def test_state():
    st = pcvl.AnnotatedBasicState([1, 0])
    assert str(st) == "|1,0>"
    assert st.n == 1
    assert st.has_annotations is False


def test_tensor_product_0():
    st1 = pcvl.AnnotatedBasicState([0, 1])
    st2 = pcvl.AnnotatedBasicState([1])
    assert str(st1*st2) == "|0,1,1>"


def test_tensor_product_1():
    st1 = pcvl.AnnotatedBasicState([1, 2])
    st2 = pcvl.AnnotatedBasicState([3, 4])
    assert str(st1*st2) == "|1,2,3,4>"


def test_tensor_product_2():
    st1 = pcvl.AnnotatedBasicState([1, 2], {2: {"P": "V"}})
    assert str(st1) == "|1,{P:V}1>"
    st2 = pcvl.AnnotatedBasicState([3, 4], {4: {"P": "H"}})
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
    st = pcvl.AnnotatedBasicState("|0,1,2>", {2: {"P": "V"}})
    assert st.n == 3
    assert st.m == 3
    assert st.has_annotations
    assert str(st) == '|0,1,{P:V}1>'
    assert st.get_mode_annotations(1) == ({},)
    assert st.get_mode_annotations(2) == ({'P': 'V'}, {})
    assert st.get_photon_annotations(2) == {'P': 'V'}
    assert st.get_photon_annotations(3) == {}
    st.set_photon_annotations(3, {"P": "H"})
    assert str(st) == '|0,1,{P:H}{P:V}>'
    st.set_photon_annotations(3, {"P": "V"})
    assert str(st) == '|0,1,2{P:V}>'


def test_state_has_annots():
    st1 = pcvl.AnnotatedBasicState("|0,1,2>", {2: {"P": "V"}})
    st2 = pcvl.AnnotatedBasicState("|0,1,2>", {3: {"P": "V"}})
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
    assert strip_line_12(svd.pdisplay()) == strip_line_12("""
            +--------+-------------+
            | state  | probability |
            +--------+-------------+
            | |0,1>  |     1/2     |
            | |1,0>  |     1/2     |
            +--------+-------------+""")


def test_sv_separation_0():
    st1 = pcvl.AnnotatedBasicState("|0,0>")
    assert st1.separate_state() == [pcvl.BasicState("|0,0>")]


def test_sv_separation_1():
    st1 = pcvl.AnnotatedBasicState("|0,1>")
    assert st1.separate_state() == [pcvl.BasicState("|0,1>")]
    st2 = pcvl.AnnotatedBasicState("|2,1>")
    assert st2.separate_state() == [pcvl.BasicState("|2,1>")]


def test_sv_separation_2():
    st1 = pcvl.AnnotatedBasicState("|0,1>", {1: {"_": 1}})
    assert st1.separate_state() == [pcvl.BasicState("|0,1>")]
    st2 = pcvl.AnnotatedBasicState("|1,1>", {1: {"_": 1}, 2: {"P": "V"}})
    assert st2.separate_state() == [pcvl.BasicState("|1,1>")]
    st3 = pcvl.AnnotatedBasicState("|1,1>", {1: {"_": 1}, 2: {"_": 1}})
    assert st3.separate_state() == [pcvl.BasicState("|1,1>")]


def test_sv_separation_3():
    st1 = pcvl.AnnotatedBasicState("|1,1>", {1: {"_": 1}, 2: {"_": 0}})
    assert st1.separate_state() == [pcvl.BasicState("|1,0>"), pcvl.BasicState("|0,1>")]
    st2 = pcvl.AnnotatedBasicState("|1,1,1>", {1: {"_": 0}, 2: {"_": 0}, 3: {"_": 1}})
    assert st2.separate_state() == [pcvl.BasicState("|1,1,0>"), pcvl.BasicState("|0,0,1>")]


def test_sv_split():
    st1 = pcvl.AnnotatedBasicState("|1,1,1>")
    partition = st1.partition([2, 1])
    expected = ["|1,1,0> |0,0,1>", "|1,0,1> |0,1,0>", "|0,1,1> |1,0,0>"]
    result = []
    for a_subset in partition:
        result.append(" ".join([str(state) for state in a_subset]))
    for r, e in zip(sorted(result), sorted(expected)):
        assert r == e


def test_sv_parse_annot():
    invalid_str = ["|{_:0}", "|{_:0},>", "|{_:0},,1>", "|{_:0},>", "|0{_:0}>", "|1{_:2>", "{P:(0.3,0)>",
                   "|{;}>", "|{P:(1,2,3)}>", "|{P:(1,a)}>", "|{a:0,a:1}>"]
    for s in invalid_str:
        with pytest.raises(ValueError):
            pcvl.AnnotatedBasicState(s)
    st1 = pcvl.AnnotatedBasicState("|{_:0}{_:1}>")
    assert str(st1.clear()) == "|2>"
    st1 = pcvl.AnnotatedBasicState("|{_:0}{_:1},0,1>")
    assert str(st1.clear()) == "|2,0,1>"
    st1 = pcvl.AnnotatedBasicState("|{_:ab,p:cd}{_:1},2>")
    assert str(st1) == "|{_:1}{_:ab,p:cd},2>"


def test_sv_parse_symb_annot():
    st1 = pcvl.AnnotatedBasicState("|{P:pi*1/2}>")
    assert str(st1) == "|{P:D}>"


def test_sv_parse_tuple_annot():
    st1 = pcvl.AnnotatedBasicState("|{P:(0.30,0)}>")
    assert str(st1) == "|{P:0.3}>"
    st1 = pcvl.AnnotatedBasicState("|{P:(pi/2,0.3)}>")
    assert str(st1) == "|{P:(pi/2,0.3)}>"


def test_sv_sample():
    source = pcvl.Source(brightness=1, purity=0.9, indistinguishability=0.9)
    qpu = pcvl.Processor({0: source, 1: source}, symb.BS())
    sample = qpu.source_distribution.sample(1)
    assert isinstance(sample, pcvl.StateVector)
    sample = qpu.source_distribution.sample(2)
    assert len(sample) == 2
    assert isinstance(sample[0], pcvl.StateVector)
    assert isinstance(sample[1], pcvl.StateVector)
