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
import math
import pytest

from pathlib import Path
from unittest.mock import MagicMock

import perceval as pcvl
from perceval.utils import StateVector, SVDistribution
from perceval.utils.logging import channel
from perceval.rendering import Format
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


def assert_svd_close(lhsvd, rhsvd):
    lhsvd.normalize()
    rhsvd.normalize()
    assert len(lhsvd) == len(rhsvd), f"len are different, {len(lhsvd)} vs {len(rhsvd)}"

    for lh_sv in lhsvd.keys():
        found_in_rh = False
        for rh_sv in rhsvd.keys():
            if not check_sv_close(lh_sv.__copy__(), rh_sv.__copy__()):
                continue
            found_in_rh = True
            assert pytest.approx(lhsvd[lh_sv]) == rhsvd[rh_sv], \
                f"different probabilities for {lh_sv}, {lhsvd[lh_sv]} vs {rhsvd[rh_sv]}"
            break
        assert found_in_rh, f"sv not found {lh_sv}"


def assert_bsd_close(lhbsd, rhbsd):
    assert len(lhbsd) == len(rhbsd), f"len are different, {len(lhbsd)} vs {len(rhbsd)}"

    for lh_bs in lhbsd.keys():
        if lh_bs not in rhbsd:
            assert False, f"bs not found {lh_bs}"
        assert pytest.approx(lhbsd[lh_bs]) == rhbsd[lh_bs], \
            f"different probabilities for {lh_bs}, {lhbsd[lh_bs]} vs {rhbsd[lh_bs]}"


def assert_bsd_close_enough(ref_bsd: pcvl.BSDistribution, bsd_to_check: pcvl.BSDistribution, threshold=0.8):
    for lh_bs in ref_bsd.keys():
        if ref_bsd[lh_bs] > threshold:
            assert lh_bs in bsd_to_check
            assert bsd_to_check[lh_bs] > threshold
            return
    assert False, f"No state has probability above {threshold} for bsd {ref_bsd}"


def assert_bsc_close_enough(ref_bsc: pcvl.BSCount, bsc_to_check: pcvl.BSCount):
    total = ref_bsc.total()
    assert total == bsc_to_check.total()
    threshold = math.sqrt(total)
    for lh_bs in ref_bsc.keys():
        if ref_bsc[lh_bs] < 5:
            continue
        assert lh_bs in bsc_to_check
        assert ref_bsc[lh_bs] == pytest.approx(bsc_to_check[lh_bs], ref_bsc[lh_bs]/threshold)


def  dict2svd(d: dict):
    return SVDistribution({StateVector(k): v for k, v in d.items()})


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
        pcvl.pdisplay_to_file(c, img_path, output_format=Format.MPLOT)
    elif isinstance(c, pcvl.AComponent) or isinstance(c, pcvl.components.ACircuit) or isinstance(c, pcvl.AProcessor):
        pcvl.pdisplay_to_file(c, img_path, output_format=Format.MPLOT,
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
        elif isinstance(c, pcvl.AComponent) or isinstance(c, pcvl.components.ACircuit) or isinstance(c, pcvl.AProcessor):
            ok, msg = _check_circuit(img_path, TEST_IMG_DIR /
                                     Path(circuit_name + ".svg"))
        else:
            raise NotImplementedError(f"_save_or_check not implemented for {type(c)}")
        assert ok, msg


class LogChecker():
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
