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

import re
import pytest
import time
import warnings

from functools import wraps
from pathlib import Path
from unittest.mock import MagicMock

from perceval import BasicState, Parameter, Expression
from perceval.components import ACircuit, PS, IDetector, AFFConfigurator, FFConfigurator, FFCircuitProvider, \
    BSLayeredPPNR, Detector, Experiment, Circuit, AProcessor, AComponent
from perceval.components.abstract_component import AParametrizedComponent
from perceval.utils import StateVector, SVDistribution
from perceval.utils.logging import channel
from perceval.rendering import Format, pdisplay_to_file
from perceval.rendering.circuit import ASkin, PhysSkin
from perceval.algorithm import AProcessTomography


TEST_IMG_DIR = Path(__file__).resolve().parent / 'imgs'


def strip_line_12(s: str) -> str:
    return s.strip().replace("            ", "")


def check_sv_close(sv1: StateVector, sv2: StateVector) -> bool:
    sv1.normalize()
    sv2.normalize()
    if len(sv1) != len(sv2):
        return False

    for s in sv1.keys():
        if s not in sv2:
            return False
        if sv1[s] != pytest.approx(sv2[s]):
            return False
    return True

def assert_sv_close(sv1: StateVector, sv2: StateVector):
    sv1.normalize()
    sv2.normalize()
    assert len(sv1) == len(sv2), f"{sv1} != {sv2} (len)"

    for s in sv1.keys():
        assert s in sv2, f"{sv1} != {sv2} ({s} missing from the latter)"
        assert sv1[s] == pytest.approx(sv2[s]), f"{sv1} != {sv2} (amplitudes {sv1[s]} != {sv2[s]})"


def compact_svd(svd: SVDistribution) -> SVDistribution:
    res = SVDistribution()

    for sv, prob in svd.items():
        found = False
        for res_sv in res:
            if check_sv_close(sv.__copy__(), res_sv.__copy__()):
                res[res_sv] += prob
                found = True
                break

        if not found:
            res[sv] = prob

    return res


def assert_svd_close(lhsvd, rhsvd):
    lhsvd = compact_svd(lhsvd)
    rhsvd = compact_svd(rhsvd)

    lhsvd.normalize()
    rhsvd.normalize()
    assert len(lhsvd) == len(rhsvd), f"len are different, {len(lhsvd)} vs {len(rhsvd)}"

    for lh_sv, lh_p in lhsvd.items():
        found_in_rh = False
        for rh_sv, rh_p in rhsvd.items():
            if not check_sv_close(lh_sv.__copy__(), rh_sv.__copy__()):
                continue
            found_in_rh = True
            assert pytest.approx(lh_p) == rh_p, f"different probabilities for {lh_sv}, {lh_p} vs {rh_p}"
            break
        assert found_in_rh, f"sv not found {lh_sv}"


def assert_bsd_close(lhbsd, rhbsd):
    lhbsd.normalize()
    rhbsd.normalize()
    assert len(lhbsd) == len(rhbsd), f"len are different, {len(lhbsd)} vs {len(rhbsd)}"

    for lh_bs, prob in lhbsd.items():
        assert rhbsd.get(lh_bs, 0) == pytest.approx(prob), f"different probabilities for {lh_bs}, {prob} vs {rhbsd[lh_bs]}"


def assert_circuits_eq(c_a: Circuit, c_b: Circuit):
    assert c_a.ncomponents() == c_b.ncomponents()
    assert_component_list_eq(c_a._components, c_b._components)


def assert_component_list_eq(comp_a, comp_b):
    assert len(comp_a) == len(comp_b)
    for (input_idx, input_comp), (output_idx, output_comp) in zip(comp_a, comp_b):
        assert isinstance(input_comp, type(output_comp))
        assert list(input_idx) == list(output_idx)

        if isinstance(input_comp, AParametrizedComponent):
            for param_name, param in input_comp.vars.items():
                assert param_name in output_comp.vars
                if isinstance(param, Expression):
                    assert sorted([str(p) for p in param._params]) == sorted([str(p) for p in output_comp.vars[param_name]._params])
                elif isinstance(param, Parameter):
                    other_param = output_comp.vars[param_name]
                    assert param.name == other_param.name
                    assert param.fixed == other_param.fixed
                    assert param.defined == other_param.defined
                    assert param.max == other_param.max
                    assert param.min == other_param.min
                    assert param.is_periodic == other_param.is_periodic
                    if param.defined:
                        assert float(param) == float(other_param)

        if isinstance(input_comp, AFFConfigurator):
            assert input_comp.name == output_comp.name
            assert input_comp._blocked_circuit_size == output_comp._blocked_circuit_size
            assert input_comp._offset == output_comp._offset
            assert_circuits_eq(input_comp.default_circuit, output_comp.default_circuit)
            if isinstance(input_comp, FFConfigurator):
                assert input_comp._configs == output_comp._configs
            elif isinstance(input_comp, FFCircuitProvider):
                for state, coe in input_comp._map.items():
                    assert state in output_comp._map
                    if isinstance(coe, ACircuit):
                        assert_circuits_eq(coe, output_comp._map[state])
                    else:
                        assert_experiment_equals(coe, output_comp._map[state])

        elif isinstance(input_comp, IDetector):
            assert_detector_eq(input_comp, output_comp)

        elif isinstance(input_comp, PS) or not isinstance(input_comp, ACircuit):
            assert input_comp.describe() == output_comp.describe()

        elif input_comp.defined and output_comp.defined:
            assert (input_comp.compute_unitary() == output_comp.compute_unitary()).all()

        else:
            assert input_comp.U == output_comp.U


def assert_detector_list_eq(detect_a: list[IDetector], detect_b: list[IDetector]):
    assert len(detect_a) == len(detect_b)
    for da, db in zip(detect_a, detect_b):
        assert isinstance(da, type(db))
        if da is not None:
            assert_detector_eq(da, db)


def assert_detector_eq(left_detector: IDetector, right_detector: IDetector):
    assert left_detector.name == right_detector.name
    if isinstance(left_detector, BSLayeredPPNR):
        assert left_detector._r == right_detector._r
        assert left_detector._layers == right_detector._layers
    elif isinstance(left_detector, Detector):
        assert left_detector.max_detections == right_detector.max_detections
        assert left_detector._wires == right_detector._wires


def assert_experiment_equals(experiment1: Experiment, experiment2: Experiment):
    assert experiment1.m == experiment2.m
    assert experiment1.name == experiment2.name
    assert experiment1.noise == experiment2.noise
    assert experiment1.input_state == experiment2.input_state
    assert experiment1.min_photons_filter == experiment2.min_photons_filter
    assert_component_list_eq(experiment1.components, experiment2.components)
    assert experiment1.in_port_names == experiment2.in_port_names
    assert experiment1.out_port_names == experiment2.out_port_names
    assert experiment1.post_select_fn == experiment2.post_select_fn
    assert_detector_list_eq(experiment1.detectors, experiment2.detectors)
    assert experiment1.heralds == experiment2.heralds
    assert experiment1.is_unitary == experiment2.is_unitary
    assert experiment1.has_td == experiment2.has_td
    assert experiment1.has_feedforward == experiment2.has_feedforward


def dict2svd(d: dict):
    return SVDistribution({StateVector(BasicState(k)): v for k, v in d.items()})


@pytest.fixture(scope="session")
def save_figs(pytestconfig):
    return pytestconfig.getoption("save_figs")


def _norm(svg):
    svg = svg.replace(" \n", "\n")
    svg = re.sub(r'url\(#.*?\)', 'url(#staticClipPath)', svg)
    svg = re.sub(r'<clipPath id=".*?">', '<clipPath id="staticClipPath">', svg)
    svg = re.sub(r'<dc:date>(.*)</dc:date>', '<dc:date></dc:date>', svg)
    svg = re.sub(r'<dc:title>(.*)</dc:title>', '<dc:title></dc:title>', svg)
    return svg


def _check_svg(test_path, ref_path, collection: str):
    with open(test_path) as f_test:
        test = _norm("".join(f_test.readlines()))
    with open(ref_path) as f_ref:
        ref = _norm("".join(f_ref.readlines()))
    m_test = re.search(rf'<g id="{collection}Collection.*?>((.|\n)*?)</g>', test)
    m_ref = re.search(rf'<g id="{collection}Collection.*?>((.|\n)*?)</g>', ref)
    if not m_test:
        return False, "cannot find patch in test"
    if not m_ref:
        return False, "cannot find patch in ref"
    m_test_patch = re.sub(r'url\(#.*?\)', "url()", m_test.group(1))
    m_ref_patch = re.sub(r'url\(#.*?\)', "url()", m_ref.group(1))
    if m_test_patch != m_ref_patch:
        return False, "test and ref are different"
    return True, "ok"


def _check_circuit(test_path, ref_path):
    return _check_svg(test_path, ref_path, 'Patch')


def _check_qpt(test_path, ref_path):
    return _check_svg(test_path, ref_path, 'Line3D')


def _save_or_check(c, tmp_path, circuit_name, save_figs, recursive=False, compact=False,
                   skin_type: type[ASkin] = PhysSkin) -> None:
    img_path = (TEST_IMG_DIR if save_figs else tmp_path) / \
        Path(circuit_name + ".svg")
    skin = skin_type(compact)

    if isinstance(c, AProcessTomography):
        pdisplay_to_file(c, img_path, output_format=Format.MPLOT)
    elif isinstance(c, (AComponent, AProcessor, Experiment)):
        pdisplay_to_file(c, img_path, output_format=Format.MPLOT,
                              recursive=recursive, skin=skin)
    else:
        raise NotImplementedError(f"_save_or_check not implemented for {type(c)}")

    if save_figs:
        with open(img_path) as f_saved:
            saved = "".join(f_saved.readlines())
        saved = _norm(saved)
        with open(img_path, "w") as fw_saved:
            fw_saved.write(saved)
    else:
        if isinstance(c, AProcessTomography):
            ok, msg = _check_qpt(img_path, TEST_IMG_DIR /
                                 Path(circuit_name + ".svg"))
        elif isinstance(c, (AComponent, AProcessor, Experiment)):
            ok, msg = _check_circuit(img_path, TEST_IMG_DIR /
                                     Path(circuit_name + ".svg"))
        else:
            raise NotImplementedError(f"_save_or_check not implemented for {type(c)}")
        assert ok, msg


class LogChecker:
    """Check if the logger as the log the expected number of log.

    The syntax to use this class is:

    with LogChecker(mock):
        do_something_that_expect_log

    It is inspired of the method pytest.warns()

    :param mock_warn: Mock object of the logger method to check (can be logger.warn, .info, .err ...)
    :param log_channel: Channel of the log to check, defaults to channel.user
    :param expected_log_number: Number of log expected to check, defaults to 1
    """
    def __init__(self, mock_warn: MagicMock, log_channel: channel = channel.user, expected_log_number: int = 1):

        self._mock_warn = mock_warn
        self._call_count = mock_warn.call_count
        self._expected_channel = log_channel
        self._expected_log_number = expected_log_number

    def __enter__(self):
        """Begging of the with statement.

        Check that no message has been log since last with statement.

        :return: self
        """
        assert self._mock_warn.call_count == self._call_count
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End of the with statement

        Check within the with statement that:
            - expected_log_number message(s) have been log/
            Then for each messages that:
                - first argument is a str
                - second argument is the expected log channel

        :param exc_type: _
        :param exc_val: _
        :param exc_tb: _
        """
        self._call_count += self._expected_log_number
        assert self._mock_warn.call_count == self._call_count, f"logger.warn: expected {self._call_count} log but got {self._mock_warn.call_count}"
        for i in range(self._expected_log_number):
            assert isinstance(self._mock_warn.mock_calls[-i].args[0], str)
            assert self._mock_warn.mock_calls[-i].args[1] == self._expected_channel

    def set_expected_log_number(self, expected_log_number: int):
        """Setter of how many log message are expected

        :param expected_log_number: expected log number
        """
        self._expected_log_number = expected_log_number


def retry(exception_to_check: type[Exception], tries: int = 4, delay: float = 0, backoff: float = 1, logger=None):
    """Retry calling the decorated function using an exponential backoff.

    http://www.saltycrane.com/blog/2009/11/trying-out-retry-decorator-python/
    original from: http://wiki.python.org/moin/PythonDecoratorLibrary#Retry

    :param exception_to_check: the exception type(s) to check; may be a tuple of types
    :param tries: number of tries (not retries) before giving up
    :param delay: initial delay between retries in seconds
    :param backoff: backoff multiplier e.g. value of 2 will double the delay each retry
    :param logger: logger to use. If None, send a warning
    """
    def deco_retry(f):

        @wraps(f)
        def f_retry(*args, **kwargs):
            mtries, mdelay = tries, delay
            while mtries > 1:
                try:
                    return f(*args, **kwargs)
                except exception_to_check as e:
                    msg = f"{e}, Retrying in {mdelay} seconds..."
                    if logger:
                        logger.warning(msg)
                    else:
                        warnings.warn(msg)
                    time.sleep(mdelay)
                    mtries -= 1
                    mdelay *= backoff
            return f(*args, **kwargs)

        return f_retry  # true decorator

    return deco_retry
