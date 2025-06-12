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
import numpy as np
import pytest
from unittest.mock import patch

import perceval as pcvl
from perceval import BSDistribution
from perceval.components import Circuit, Processor, BS, PS, catalog, UnavailableModeException, Port, PortLocation, \
    PERM, Detector
from perceval.utils import BasicState, StateVector, SVDistribution, Encoding, NoiseModel, P
from perceval.backends import Clifford2017Backend

from _test_utils import LogChecker, assert_svd_close, assert_bsd_close


@patch.object(pcvl.utils.logging.ExqaliburLogger, "warn")
def test_processor_input_fock_state(mock_warn):
    p = Processor("Naive", Circuit(4))  # Init with perfect source
    with LogChecker(mock_warn, expected_log_number=0):
        p.with_input(BasicState([0, 1, 1, 0]))
    assert_svd_close(p.source_distribution, SVDistribution({StateVector([0, 1, 1, 0]): 1}))


def test_processor_input_fock_state_with_loss():
    p = Processor("Naive", Circuit(4), NoiseModel(brightness=0.2))
    p.with_input(BasicState([0, 1, 1, 0]))
    expected = {
        StateVector([0, 1, 1, 0]): 0.04,
        StateVector([0, 1, 0, 0]): 0.16,
        StateVector([0, 0, 1, 0]): 0.16,
        StateVector([0, 0, 0, 0]): 0.64
    }
    assert_svd_close(p.source_distribution, expected)


def test_processor_input_fock_state_with_all_noise_sources():
    nm = NoiseModel(brightness=0.2, indistinguishability=0.9, g2=0.1, g2_distinguishable=False)
    p = Processor("Naive", Circuit(4), nm)
    p.source.simplify_distribution = True
    p.with_input(BasicState([0, 1, 1, 0]))

    expected = {'|0,0,0,0>': 16 / 25,
                '|0,0,2{_:0},0>': 0.0015490319977879558,
                '|0,0,{_:0},0>': 0.15836717690616972,
                '|0,0,{_:0}{_:1},0>': 8.37910960423266e-05,
                '|0,2{_:0},0,0>': 0.0015490319977879558,
                '|0,2{_:0},2{_:0},0>': 3.749218953392102e-06,
                '|0,2{_:0},{_:0},0>': 0.0003636359771584214,
                '|0,2{_:0},{_:0}{_:1},0>': 2.0280482640513694e-07,
                '|0,2{_:0},{_:1},0>': 1.96699985087703e-05,
                '|0,{_:0},0,0>': 0.15836717690616972,
                '|0,{_:0},2{_:0},0>': 0.0003636359771584214,
                '|0,{_:0},2{_:1},0>': 1.96699985087703e-05,
                '|0,{_:0},{_:0},0>': 0.03526897882672976,
                '|0,{_:0},{_:0}{_:1},0>': 1.96699985087703e-05,
                '|0,{_:0},{_:1},0>': 0.0039187754251921985,
                '|0,{_:0},{_:1}{_:2},0>': 1.0640004445062523e-06,
                '|0,{_:0}{_:1},0,0>': 8.37910960423266e-05,
                '|0,{_:0}{_:1},2{_:0},0>': 2.0280482640513694e-07,
                '|0,{_:0}{_:1},{_:0},0>': 1.96699985087703e-05,
                '|0,{_:0}{_:1},{_:0}{_:2},0>': 1.097023089996e-08,
                '|0,{_:0}{_:1},{_:2},0>': 1.0640004445062523e-06}
    result = {str(k): v for k, v in p.source_distribution.items()}
    assert pytest.approx(expected) == result
    assert pytest.approx(sum([v for v in p.source_distribution.values()])) == 1


def test_processor_input_state_vector():
    p = Processor("Naive", Circuit(4))  # Init with perfect source
    sv = BasicState([0, 1, 1, 0]) + BasicState([1, 0, 0, 1])
    p.with_input(sv)
    assert_svd_close(p.source_distribution, {sv: 1})

    p = Processor("Naive", Circuit(4), noise=NoiseModel(transmittance=.4, g2=.06))  # Init with noise
    sv = BasicState([0, 1, 1, 0]) + BasicState([1, 0, 0, 1])
    p.with_input(sv)
    assert_svd_close(p.source_distribution, {sv: 1})  # The source does NOT affect SV inputs


def test_processor_probs():
    qpu = Processor("Naive", BS())
    qpu.with_input(BasicState([1, 1]))  # Are expected only states with 2 photons in the same mode.
    for m in range(qpu.circuit_size):
        qpu.add(m, Detector.threshold())  # With threshold detectors, the simulation will only detect |1,0> and |0,1>
    qpu.min_detected_photons_filter(2)
    probs = qpu.probs()

    # By default, all states are filtered and physical performance drops to 0
    assert pytest.approx(probs['global_perf']) == 0

    detectors = qpu.detectors
    for i in range(len(detectors)):
        detectors[i] = None  # Ugly reset of detectors

    # With perfect detection, we get our results back
    probs = qpu.probs()
    bsd_out = probs['results']
    assert pytest.approx(bsd_out[BasicState("|2,0>")]) == 0.5
    assert pytest.approx(bsd_out[BasicState("|0,2>")]) == 0.5
    assert pytest.approx(probs['global_perf']) == 1


def test_processor_samples():
    proc = Processor(Clifford2017Backend(), BS())

    # Without annotations
    proc.with_input(BasicState("|1,1>"))
    samples = proc.samples(500)
    assert samples["results"].count(BasicState([1, 1])) == 0
    assert len(samples["results"]) == 500

    # With annotations
    proc.with_input(SVDistribution({BasicState("|{_:0},{_:1}>"): 1}))
    samples = proc.samples(500)
    assert samples["results"].count(BasicState([1, 1])) > 50


def test_processor_samples_max_shots():
    p = Processor(Clifford2017Backend(), 4)  # Identity circuit with perfect source
    p.with_input(BasicState([1, 1, 1, 1]))
    for (max_samples, max_shots) in [(10, 10), (2, 50), (17, 2), (0, 11), (10, 0)]:
        # In this trivial case, 1 shot = 1 sample, so test this pair of parameters work as a dual threshold system
        samples = p.samples(max_samples, max_shots)
        assert len(samples['results']) == min(max_samples, max_shots)

    p = Processor(Clifford2017Backend(), 4)
    p.add(0, catalog['postprocessed cnot'].build_processor())
    p.with_input(BasicState([0, 1, 0, 1]))
    max_samples = 100
    result_len = {}
    for max_shots in [100, 500, 2_000]:  # Success prob = 1/9  --> AVG 900 shots to get 100 samples
        result_len[max_shots] = len(p.samples(max_samples, max_shots)['results'])
    assert result_len[100] < result_len[500]
    assert result_len[500] < result_len[2_000]
    assert result_len[2_000] == max_samples  # 2k shots is enough to get the expected sample count


def test_add_remove_ports():
    processor = Processor("SLOS", 6)
    p0 = Port(Encoding.DUAL_RAIL, "q0")
    p1 = Port(Encoding.DUAL_RAIL, "q1")
    p2 = Port(Encoding.DUAL_RAIL, "q2")
    processor.add_port(0, p0, PortLocation.OUTPUT)
    processor.add_port(2, p1)
    processor.add_port(4, p2, PortLocation.INPUT)

    with pytest.raises(UnavailableModeException):
        processor.add_port(4, p2, PortLocation.INPUT)

    assert processor.in_port_names == ["", "", "q1", "q1", "q2", "q2"]
    assert processor.out_port_names == ["q0", "q0", "q1", "q1", "", ""]

    assert processor.get_input_port(0) is None
    assert processor.get_input_port(1) is None
    assert processor.get_input_port(2) is p1
    assert processor.get_input_port(3) is p1
    assert processor.get_input_port(4) is p2
    assert processor.get_input_port(5) is p2

    assert processor.get_output_port(0) is p0
    assert processor.get_output_port(1) is p0
    assert processor.get_output_port(2) is p1
    assert processor.get_output_port(3) is p1
    assert processor.get_output_port(4) is None
    assert processor.get_output_port(5) is None

    processor.remove_port(2, PortLocation.OUTPUT)

    with pytest.raises(UnavailableModeException):
        processor.remove_port(2, PortLocation.OUTPUT)

    with pytest.raises(UnavailableModeException):
        processor.add_port(2, p1)

    assert processor.in_port_names == ["", "", "q1", "q1", "q2", "q2"]
    assert processor.out_port_names == ["q0", "q0", "", "", "", ""]

    assert processor.get_input_port(0) is None
    assert processor.get_input_port(1) is None
    assert processor.get_input_port(2) is p1
    assert processor.get_input_port(3) is p1
    assert processor.get_input_port(4) is p2
    assert processor.get_input_port(5) is p2

    assert processor.get_output_port(0) is p0
    assert processor.get_output_port(1) is p0
    assert processor.get_output_port(2) is None
    assert processor.get_output_port(3) is None
    assert processor.get_output_port(4) is None
    assert processor.get_output_port(5) is None

    processor.remove_port(0, PortLocation.OUTPUT)
    processor.remove_port(2, PortLocation.INPUT)
    processor.remove_port(4, PortLocation.INPUT)

    with pytest.raises(UnavailableModeException):
        processor.remove_port(2, PortLocation.INPUT)

    for i in range(6):
        assert processor.get_input_port(i) is None
        assert processor.get_output_port(i) is None


def test_phase_quantization():
    nm = NoiseModel(phase_imprecision=0.1)
    p0 = Processor("SLOS", catalog["mzi phase first"].build_circuit(phi_a=0.596898191919898198,
                                                                    phi_b=0.16561561651616))
    p1 = Processor("SLOS", catalog["mzi phase first"].build_circuit(phi_a=0.596898191919898198,
                                                                    phi_b=0.16561561651616), noise=nm)
    p2 = Processor("SLOS", catalog["mzi phase first"].build_circuit(phi_a=0.6,
                                                                    phi_b=0.2))

    p0.with_input(BasicState([1, 1]))
    p1.with_input(BasicState([1, 1]))
    p2.with_input(BasicState([1, 1]))
    with pytest.raises(AssertionError):
        assert_bsd_close(p0.probs()["results"], p1.probs()["results"])
    assert_bsd_close(p1.probs()["results"], p2.probs()["results"])

    p1.noise = NoiseModel()
    assert_bsd_close(p0.probs()["results"], p1.probs()["results"])

    p0.noise = nm
    p1.noise = nm
    assert_bsd_close(p0.probs()["results"], p1.probs()["results"])


def test_phase_error():
    error: float = 0.2
    angle: float = 1

    nm = NoiseModel(phase_error=error)
    p = Processor("SLOS", PS(phi=angle), noise=nm)
    c = p.linear_circuit()
    assert float(c._components[0][1].param("max_error")) == pytest.approx(error)
    a = c.compute_unitary()[0, 0]
    assert np.angle(a) != angle
    assert angle-error <= np.angle(a) <= angle+error

    # When a specific error is directly set on a phase shifter, the NoiseModel does not change the value
    specific_error: float = 0.3
    p = Processor("SLOS", PS(phi=angle, max_error=specific_error), noise=nm)
    c = p.linear_circuit()
    assert float(c._components[0][1].param("max_error")) == pytest.approx(specific_error)
    a = c.compute_unitary()[0, 0]
    assert np.angle(a) != angle
    assert angle - specific_error <= np.angle(a) <= angle + specific_error

    # Works with symbols too
    symb_error = "err0"
    symb_angle = "phi0"
    p = Processor("SLOS", PS(phi=P(symb_angle), max_error=P(symb_error)), noise=nm)
    c = p.linear_circuit()
    err_param = c._components[0][1].param("max_error")
    assert err_param.is_variable and str(err_param.spv) == symb_error
    a = c.compute_unitary(use_symbolic=True)[0, 0]
    assert 'exp(I*' in str(a) and '*err0 + phi0' in str(a)


def test_empty_output():
    p = Processor("SLOS", 4)
    p.add(0, PERM([1, 0]))
    p.add_herald(0, 1)
    p.min_detected_photons_filter(2)
    p.with_input(BasicState([0, 1, 0]))

    res = p.probs()["results"]
    assert res == BSDistribution()


def test_mask_distinguishability():
    p = Processor("SLOS", 3, noise=NoiseModel(indistinguishability=.5))

    p.add_herald(1, 1)
    p.add_herald(2, 1)

    p.min_detected_photons_filter(0)
    input_state = BasicState([0])
    p.with_input(input_state)

    assert p.probs()["results"][input_state] == pytest.approx(1)


def test_mask_detectors():
    p = Processor("SLOS", 2)

    p.add(0, BS())
    p.add(0, Detector.threshold())
    p.add_herald(0, 1)  # Since using a threshold, expected is at least 1

    p.min_detected_photons_filter(0)
    p.with_input(BasicState([1]))

    res = p.probs()

    assert res["results"][BasicState([0])] == pytest.approx(1)
    assert res["global_perf"] == pytest.approx(.5)


def test_min_photons_reset():
    p = Processor("SLOS", 2, noise=NoiseModel(brightness=.5))

    p.min_detected_photons_filter(2)
    input_state = BasicState([1, 1])
    p.with_input(input_state)

    res = p.probs()
    assert res["results"][input_state] == pytest.approx(1)
    assert res["global_perf"] == pytest.approx(.25)

    p.min_detected_photons_filter(0)
    res = p.probs()
    assert res["results"][input_state] == pytest.approx(.25)
    assert res["global_perf"] == pytest.approx(1)


def test_flatten_processor():
    ansatz = catalog["qloq ansatz"]
    group_sizes = [Encoding.QUDIT2, Encoding.QUDIT2]
    layers = ["Y"]
    phases = None
    p = ansatz.build_processor(
        group_sizes=group_sizes,
        layers=layers,
        phases=phases,
        ctype="cz"
    )

    level_0_components = p.flatten(0)
    level_1_components = p.flatten(1)
    level_2_components = p.flatten(2)
    level_3_components = p.flatten(3)
    level_4_components = p.flatten(4)
    all_components = p.flatten()

    assert len(level_0_components) == 1 # 1 sub-circuit
    assert len(level_1_components) == 15 # 13 sub-sub-circuits (8 RYQUDIT2, 4 CZ2, 1 POSTPROCESSED CZ) + 2 PERM
    assert len(level_2_components) == 60 # 8*4 in RYQUDIT2, 4*5 in CZ2, 1*6 in POSTPROCESSED CZ + 2 PERM
    assert len(level_3_components) == 68 # 8*5 in RYQUDIT2, 4*5 in CZ2, 1*6 in POSTPROCESSED CZ + 2 PERM
    assert len(level_4_components) == 76 # 8*6 in RYQUDIT2, 4*5 in CZ2, 1*6 in POSTPROCESSED CZ + 2 PERM
    assert len(all_components) == len(level_4_components)


def test_asymmetric_processor():
    # Herald at left
    p = Processor("SLOS", 2)
    p.add(0, BS())
    p.add_herald(0, 1, location=PortLocation.INPUT)

    assert p.m == 2
    assert p.m_in == 1

    assert p.heralds == {}
    assert p.in_heralds == {0: 1}

    p.with_input(BasicState([0]))
    assert p.input_state == BasicState([1, 0])

    res = p.probs()
    assert res["results"][BasicState([1, 0])] == pytest.approx(0.5)
    assert res["results"][BasicState([0, 1])] == pytest.approx(0.5)

    # Herald at right
    p = Processor("SLOS", 2)
    p.add(0, BS())
    p.add_herald(0, 1, location=PortLocation.OUTPUT)

    assert p.m == 1
    assert p.m_in == 2

    assert p.heralds == {0: 1}
    assert p.in_heralds == {}

    p.with_input(BasicState([1, 0]))

    res = p.probs()
    assert res["results"][BasicState([0])] == pytest.approx(1)
    assert res["global_perf"] == pytest.approx(.5)
