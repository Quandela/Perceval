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

from perceval.utils import StateVector, SVDistribution

import pytest


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


def  dict2svd(d: dict):
    return SVDistribution({StateVector(k): v for k, v in d.items()})


if __name__ == "__main__":
    sv1 = StateVector([0, 1]) + StateVector([1, 0])
    sv1_bis = 1.0000001*StateVector([0, 1]) + 0.9999999*StateVector([1, 0])
    assert_sv_close(sv1, sv1_bis)
    sv2 = StateVector([0, 1]) - StateVector([1, 0])
    try:
        assert_sv_close(sv1, sv2)
    except AssertionError:
        print("detected sv are different")

    sv3 = StateVector([0, 1]) + StateVector([1, 1])
    try:
        assert_sv_close(sv1, sv3)
    except AssertionError:
        print("detected sv are different")

    sv4 = StateVector([0, 1])
    try:
        assert_sv_close(sv1, sv4)
    except AssertionError:
        print("detected sv are different")

    try:
        assert_sv_close(sv4, sv1)
    except AssertionError:
        print("detected sv are different")
