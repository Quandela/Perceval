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

from collections import Counter
import math
import platform
import pytest

import perceval as pcvl
import perceval.components.unitary_components as comp
from perceval.utils import BasicState, StateVector, SVDistribution, allstate_iterator
from perceval.rendering.pdisplay import pdisplay_state_distrib
from _test_utils import strip_line_12, assert_sv_close, assert_svd_close


def test_basic_state():
    st = BasicState([1, 0])
    assert str(st) == "|1,0>"
    assert st.n == 1
    assert st.has_annotations is False

    with pytest.raises(RuntimeError):
        bs = BasicState([0] * 300)  # 300 modes is too much (mode count is capped at 256)


def test_str_state_vector():
    sv = (1 + 1j) * StateVector("|0,1>") + (1 - 1j) * StateVector("|1,0>")
    assert str(sv) == "(0.5+0.5I)*|0,1>+(0.5-0.5I)*|1,0>" \
        or str(sv) == "(0.5-0.5I)*|1,0>+(0.5+0.5I)*|0,1>"  # Order doesn't matter


def test_tensor_product_0():
    st1 = BasicState([0, 1])
    st2 = BasicState([1])
    assert str(st1 * st2) == "|0,1,1>"


def test_tensor_product_1():
    st1 = BasicState([1, 2])
    st2 = BasicState([3, 4])
    assert str(st1 * st2) == "|1,2,3,4>"


def test_tensor_product_2():
    st1 = BasicState("|1,{P:V}1>")
    assert str(st1) == "|1,{P:V}1>"
    st2 = BasicState("|3,{P:H}3>")
    assert str(st2) == "|3,{P:H}3>"
    assert str(st1 * st2) == "|1,{P:V}1,3,{P:H}3>"


def test_tensor_svdistribution_1():
    sv1 = SVDistribution()
    sv1.add(StateVector([0]), 0.25)
    sv1.add(StateVector([1]), 0.75)
    sv2 = SVDistribution()
    sv2.add(StateVector([0]), 0.8)
    sv2.add(StateVector([1]), 0.2)
    sv = sv1 * sv2
    assert len(sv) == 4
    assert pytest.approx(sv[StateVector([0, 0])]) == 1 / 5
    assert pytest.approx(sv[StateVector([0, 1])]) == 1 / 20
    assert pytest.approx(sv[StateVector([1, 0])]) == 3 / 5
    assert pytest.approx(sv[StateVector([1, 1])]) == 3 / 20


def test_state_annots():
    st = BasicState('|0,1,{P:V}1>')
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
    st1 = BasicState("|0,1,{P:V}1>")
    st2 = BasicState("|0,1,{P:V}1>")
    assert st1 == st2

    st3 = BasicState("|{a:1}2{a:0},0,0>")
    st4 = BasicState("|2{a:0}{a:1},0,0>")
    assert st3 == st4

    sv = 0.5 * st1 + st3 - st4
    assert_sv_close(sv, StateVector(st1))


def test_state_invalid_superposition():
    st1 = StateVector("|0,1>")
    st2 = StateVector("|1,0,0>")
    with pytest.raises(RuntimeError):
        st1 + st2


def test_state_superposition():
    st1 = StateVector("|0,1>")
    st2 = StateVector("|1,0>")
    assert_sv_close(2 * st1, st1)
    assert_sv_close(st1 + 1j * st2, StateVector('|0,1>') + 1j * StateVector('|1,0>'))


def test_init_state_vector():
    st = StateVector()
    st += BasicState("|1,0>")
    assert str(st) == "|1,0>"


def test_bsdistribution():
    bs1 = BasicState([1, 2])
    bs2 = BasicState([3, 4])
    bsd = pcvl.BSDistribution()
    bsd[bs1] = 0.9
    bsd[bs2] = 0.1
    bsd_squared = bsd ** 2
    assert isinstance(bsd_squared, pcvl.BSDistribution)
    assert len(bsd_squared) == 4
    assert bsd_squared[bs1 * bs1] == pytest.approx(0.81)
    assert bsd_squared[bs1 * bs2] == pytest.approx(0.09)
    assert bsd_squared[bs2 * bs1] == pytest.approx(0.09)
    assert bsd_squared[bs2 * bs2] == pytest.approx(0.01)

    bsd_mult = pcvl.BSDistribution(bs1) * pcvl.BSDistribution({bs1: 0.4, bs2: 0.6})
    assert len(bsd_mult) == 2
    assert bsd_mult[bs1 * bs1] == pytest.approx(0.4)
    assert bsd_mult[bs1 * bs2] == pytest.approx(0.6)
    assert bsd.m == 2
    assert bsd_squared.m == 4
    with pytest.raises(ValueError):
        pcvl.BSDistribution({BasicState("|1>"): .5, BasicState("|1,1>"): .5})


def test_svdistribution():
    st1 = StateVector("|0,1>")
    st2 = StateVector("|1,0>")
    svd = SVDistribution()
    svd.add(st1, 0.5)
    svd[st2] = 0.5
    assert strip_line_12(pdisplay_state_distrib(svd)) == strip_line_12("""
            +-------+-------------+
            | state | probability |
            +-------+-------------+
            | |0,1> |     1/2     |
            | |1,0> |     1/2     |
            +-------+-------------+""")
    svd_squared = svd ** 2
    assert isinstance(svd_squared, SVDistribution)
    assert len(svd_squared) == 4
    assert svd_squared[StateVector("|1,0,1,0>")] == pytest.approx(1 / 4)
    assert svd_squared[StateVector("|1,0,0,1>")] == pytest.approx(1 / 4)
    assert svd_squared[StateVector("|0,1,1,0>")] == pytest.approx(1 / 4)
    assert svd_squared[StateVector("|0,1,0,1>")] == pytest.approx(1 / 4)
    assert svd.m == 2
    assert svd.n_max == 1
    assert svd_squared.m == 4
    assert svd_squared.n_max == 2
    with pytest.raises(ValueError):
        svd[StateVector("|1>")] = 1/7
        SVDistribution({StateVector("|1>"): .5, StateVector("|1,1>"): .5})


def test_separate_state_without_annots():
    st1 = BasicState("|0,1>")
    assert st1.separate_state() == [BasicState("|0,1>")]
    st2 = BasicState("|2,1>")
    assert st2.separate_state() == [BasicState("|2,1>")]


def test_separate_state_with_annots():
    st1 = BasicState("|0,{_:1}>")
    assert st1.separate_state(keep_annotations=True) == [st1]
    st2 = BasicState("|{_:1},{P:V}>")
    assert st2.separate_state(keep_annotations=False) == [BasicState("|1,1>")]
    st3 = BasicState("|{_:1},{_:2}>")
    assert st3.separate_state(False) == [BasicState("|1,0>"), BasicState("|0,1>")]
    assert st3.separate_state(keep_annotations=True) == [BasicState("|{_:1},0>"), BasicState("|0,{_:2}>")]
    st4 = BasicState("|{_:1},{_:0}>")
    assert st4.separate_state(keep_annotations=False) == [BasicState("|1,0>"), BasicState("|0,1>")]
    assert st4.separate_state(True) == [BasicState("|{_:1},0>"), BasicState("|0,{_:0}>")]
    st5 = BasicState("|{_:0},{_:0},{_:3}>")
    # By default, keep_annotations is false
    assert st5.separate_state() == [BasicState("|1,1,0>"), BasicState("|0,0,1>")]
    assert st5.separate_state(keep_annotations=True) == [BasicState("|{_:0},{_:0},0>"), BasicState("|0,0,{_:3}>")]


def test_partition():
    st1 = BasicState("|1,1,1>")
    partition = st1.partition([2, 1])
    expected = ["|1,1,0> |0,0,1>", "|1,0,1> |0,1,0>", "|0,1,1> |1,0,0>"]
    result = []
    for a_subset in partition:
        result.append(" ".join([str(state) for state in a_subset]))
    for r, e in zip(sorted(result), sorted(expected)):
        assert r == e


def test_parse_annot():
    invalid_str = ["|{_:0}", "|0{_:0}>", "|1{_:2>", "{P:(0.3,0)>",
                   "|{;}>", "|{P:(1,2,3)}>", "|{P:(1,a)}>", "|{a:0,a:1}>"]
    for s in invalid_str:
        with pytest.raises(ValueError):
            BasicState(s)
    st1 = BasicState("|{_:0}{_:1}>")
    st1.clear_annotations()
    assert st1 == BasicState("|2>")
    st1 = BasicState("|{_:0}{_:1},0,1>")
    st1.clear_annotations()
    assert st1 == BasicState("|2,0,1>")


def test_sv_parse_symb_annot():
    st1 = BasicState("|{P:1.5707963268}>")
    assert str(st1) == "|{P:D}>"


def test_sv_parse_tuple_annot():
    st1 = BasicState("|{P:(0.30,0)}>")
    assert str(st1) == "|{P:0.3}>"
    # st1 = BasicState("|{P:(pi/2,0.3)}>")
    # assert str(st1) == "|{P:(pi/2,0.3)}>"


def test_svd_sample():
    source = pcvl.Source(emission_probability=1, multiphoton_component=0.1, indistinguishability=0.9)
    qpu = pcvl.Processor("Naive", comp.BS(), source)
    qpu.with_input(BasicState([1, 0]))
    sample = qpu.source_distribution.sample(1)
    assert isinstance(sample, list)
    assert isinstance(sample[0], StateVector)
    assert sample[0] in qpu.source_distribution
    sample = qpu.source_distribution.sample(2)
    assert len(sample) == 2
    assert isinstance(sample[0], StateVector)
    assert isinstance(sample[1], StateVector)


def test_svd_anonymize_annots_simple():
    svd = SVDistribution({
        BasicState("|{_:1},{_:2},{_:3}>"): 0.2,
        BasicState("|{_:4},{_:5},{_:6}>"): 0.3,
        BasicState("|{_:1},{_:3},{_:3}>"): 0.4,
        BasicState("|{_:0},{_:8},{_:8}>"): 0.1,
    })
    svd2 = pcvl.anonymize_annotations(svd)
    assert_svd_close(svd2, SVDistribution(
        {
            StateVector('|{a:0},{a:1},{a:2}>'): 0.5,
            StateVector('|{a:0},{a:1},{a:1}>'): 0.5
        }))


@pytest.mark.skipif(
    platform.system() != "Windows",
    reason="Superposed states aren't anonymized the same given C++ unordered containers OS-specific implementation")
def test_svd_anonymize_annots_superposition():
    svd = SVDistribution({
        StateVector("|{_:0},{_:0},{_:1}>") + StateVector("|{_:0},{_:2},{_:1}>"): 0.1,
        StateVector("|{_:3},{_:4},{_:4}>"): 0.1,
        StateVector("|{_:0},{_:0},{_:1}>") + StateVector("|{_:0},{_:4},{_:1}>"): 0.1,
        BasicState("|{_:4},{_:4},{_:2}>"): 0.1,
        BasicState("|{_:1},{_:3},{_:3}>"): 0.1,
        BasicState("|{_:0},{_:2},{_:3}>"): 0.5,
    })
    svd2 = pcvl.anonymize_annotations(svd)
    assert_svd_close(svd2, SVDistribution(
        {
            StateVector('|{a:0},{a:1},{a:2}>'): 0.5,
            StateVector('|{a:0},{a:0},{a:1}>') + StateVector('|{a:0},{a:2},{a:1}>'): 0.2,
            StateVector('|{a:0},{a:1},{a:1}>'): 0.2,
            StateVector('|{a:0},{a:0},{a:1}>'): 0.1
        }))


def test_statevector_sample():
    sv = StateVector("|0,1>") + StateVector("|1,0>")
    counter = Counter()
    for _ in range(20):
        counter[sv.sample()] += 1
    states = [str(s) for s in counter]
    assert len(states) == 2 and "|1,0>" in states and "|0,1>" in states


def test_statevector_samples():
    sv = StateVector("|0,1>") + StateVector("|1,0>")
    counter = Counter()
    for s in sv.samples(20):
        counter[s] += 1
    states = [str(s) for s in counter]
    assert len(states) == 2 and "|1,0>" in states and "|0,1>" in states


def test_statevector_measure_1():
    sv = StateVector("|0,1>") + StateVector("|1,0>")
    map_measure_sv = sv.measure([0])
    assert len(map_measure_sv) == 2 and \
           BasicState("|0>") in map_measure_sv and \
           BasicState("|1>") in map_measure_sv
    assert pytest.approx(0.5) == map_measure_sv[BasicState("|0>")][0]
    assert map_measure_sv[BasicState("|0>")][1] == StateVector("|1>")
    assert pytest.approx(0.5) == map_measure_sv[BasicState("|1>")][0]
    assert map_measure_sv[BasicState("|1>")][1] == StateVector("|0>")


def test_statevector_measure_2():
    sv = StateVector("|0,1>") + StateVector("|1,0>")
    map_measure_sv = sv.measure([0, 1])
    assert len(map_measure_sv) == 2 and \
           BasicState("|0,1>") in map_measure_sv and \
           BasicState("|1,0>") in map_measure_sv
    assert pytest.approx(0.5) == map_measure_sv[BasicState("|0,1>")][0]
    assert map_measure_sv[BasicState("|0,1>")][1] == StateVector("|>")
    assert pytest.approx(0.5) == map_measure_sv[BasicState("|1,0>")][0]
    assert map_measure_sv[BasicState("|1,0>")][1] == StateVector("|>")


def test_statevector_measure_3():
    sv = StateVector("|0,1,1>") + StateVector("|1,1,0>")
    map_measure_sv = sv.measure([1])
    assert len(map_measure_sv) == 1 and \
           BasicState("|1>") in map_measure_sv
    assert pytest.approx(1) == map_measure_sv[BasicState("|1>")][0]
    assert_sv_close(map_measure_sv[BasicState("|1>")][1], StateVector([0, 1]) + StateVector([1, 0]))


def test_statevector_equality():
    gamma = math.pi / 2
    st1 = StateVector("|{P:H},{P:H}>")
    st2 = StateVector("|{P:H},{P:V}>")
    input_state = math.cos(gamma) * st1 + math.sin(gamma) * st2
    assert input_state == st2


def test_statevector_arithmetic():
    sv1 = StateVector()
    sv1 += StateVector([0, 1])
    sv1 += StateVector([1, 0])
    assert_sv_close(sv1, StateVector([0, 1]) + StateVector([1, 0]))

    sv2 = StateVector()
    sv2 += 0.5 * StateVector([0, 1])
    sv2 += -0.5 * StateVector([1, 0])
    assert_sv_close(sv2, StateVector([0, 1]) - StateVector([1, 0]))

    sv3 = sv1 + sv2
    assert sv3 == StateVector([0, 1])

    sv4 = 0.2j * sv1 - 0.6j * sv2
    assert_sv_close(sv4, -math.sqrt(5) / 5 * 1j * StateVector([0, 1]) + math.sqrt(5) / 5 * 2j * StateVector([1, 0]))

    sv5 = StateVector()
    sv5 += 0.2j * sv1
    sv5 += -0.6j * sv2
    assert_sv_close(sv5, -math.sqrt(5) / 5 * 1j * StateVector([0, 1]) + math.sqrt(5) / 5 * 2j * StateVector([1, 0]))


def test_max_photon_state_iterator():
    l = [[0, 0, 0],
         [1, 0, 0], [0, 1, 0], [0, 0, 1],
         [2, 0, 0], [1, 1, 0], [1, 0, 1], [0, 2, 0], [0, 1, 1], [0, 0, 2]
         ]
    i = 0
    for bs in pcvl.utils.max_photon_state_iterator(3,2):
        assert bs == pcvl.BasicState(l[i])
        i += 1


def test_allstate_iterator():
    in_bs = BasicState([1,0,0,0])
    iteration_result = list(allstate_iterator(in_bs))
    assert len(iteration_result) == 4
    assert BasicState([1, 0, 0, 0]) in iteration_result
    assert BasicState([0, 1, 0, 0]) in iteration_result
    assert BasicState([0, 0, 1, 0]) in iteration_result
    assert BasicState([0, 0, 0, 1]) in iteration_result

    in_sv = StateVector([0]+[0]*5) + StateVector([1]+[0]*5) + StateVector([2]+[0]*5)
    iteration_result = list(allstate_iterator(in_sv))
    assert len(iteration_result) == 28  # (C2,7 + C1,6 + C0,6 = 21 + 6 + 1)
