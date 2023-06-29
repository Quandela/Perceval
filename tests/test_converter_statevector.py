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
from perceval.converters import StatevectorConverter
from perceval.utils import StateVector, Encoding

has_qiskit = True
try:
    from qiskit.quantum_info import Statevector as Qiskit_sv
except ModuleNotFoundError as e:
    has_qiskit = False

has_qutip = True
try:
    from qutip import ket
except ModuleNotFoundError as e:
    has_qutip = False


@pytest.mark.skipif(not has_qiskit and not has_qutip, reason="Requires either qiskit or QuTip")
def test_convert_sv_dual_rail():
    converter = StatevectorConverter(encoding=Encoding.DUAL_RAIL, ancillae=[1, 2])
    pcvl_sv_in = StateVector('|0,2,0,1,0,1,0,1>') - StateVector('|1,0,2,0,1,0,1,0>')
    expected_sv = 1/2 * StateVector('|1,0,1,0,1,0>') \
                  - 1/2 * StateVector('|1,0,0,1,0,1>') \
                  - 1/2 * StateVector('|0,1,0,1,1,0>') \
                  - 1/2 * StateVector('|0,1,0,1,0,1>')

    if has_qiskit:
        qiskit_sv_out = converter.to_qiskit(pcvl_sv_in)
        assert qiskit_sv_out == Qiskit_sv([-0.70710678, 0., 0., 0., 0., 0., 0., 0.70710678], dims=(2, 2, 2))

        qiskit_sv = Qiskit_sv([.1, 0, 0, -.1, 0, 0, -.1, -.1])
        assert converter.to_perceval(qiskit_sv) == expected_sv

    if has_qutip:
        qutip_sv = (ket('000') - ket('011') - ket('110') - ket('111')).unit()
        assert converter.to_perceval(qutip_sv) == expected_sv


@pytest.mark.skipif(not has_qiskit and not has_qutip, reason="Requires either qiskit or QuTip")
def test_convert_sv_raw():
    converter = StatevectorConverter(encoding=Encoding.RAW, ancillae=[1, 2])
    pcvl_sv_in = StateVector('|1,2,0,1,1>') - StateVector('|0,0,2,0,0>')
    expected_sv = 1/2 * StateVector('|0,0,0>') \
                  - 1/2 * StateVector('|0,1,1>') \
                  - 1/2 * StateVector('|1,1,0>') \
                  - 1/2 * StateVector('|1,1,1>')

    if has_qiskit:
        qiskit_sv_out = converter.to_qiskit(pcvl_sv_in)
        assert qiskit_sv_out == Qiskit_sv([-0.70710678, 0., 0., 0., 0., 0., 0., 0.70710678], dims=(2, 2, 2))

        qiskit_sv = Qiskit_sv([.1, 0, 0, -.1, 0, 0, -.1, -.1])
        assert converter.to_perceval(qiskit_sv) == expected_sv

    if has_qutip:
        qutip_sv = (ket('000') - ket('011') - ket('110') - ket('111')).unit()
        assert converter.to_perceval(qutip_sv) == expected_sv


@pytest.mark.skipif(not has_qiskit and not has_qutip, reason="Requires either qiskit or QuTip")
def test_convert_sv_polarization():
    converter = StatevectorConverter(encoding=Encoding.POLARIZATION)
    pcvl_sv_in = StateVector('|{P:V},{P:V},{P:V}>') - StateVector('|{P:H},{P:H},{P:H}>')
    expected_sv = 1/2 * StateVector('|{P:H},{P:H},{P:H}>') \
                  - 1/2 * StateVector('|{P:H},{P:V},{P:V}>') \
                  - 1/2 * StateVector('|{P:V},{P:V},{P:H}>') \
                  - 1/2 * StateVector('|{P:V},{P:V},{P:V}>')

    if has_qiskit:
        qiskit_sv_out = converter.to_qiskit(pcvl_sv_in)
        assert qiskit_sv_out == Qiskit_sv([-0.70710678, 0., 0., 0., 0., 0., 0., 0.70710678], dims=(2, 2, 2))

        qiskit_sv = Qiskit_sv([.1, 0, 0, -.1, 0, 0, -.1, -.1])
        assert converter.to_perceval(qiskit_sv) == expected_sv

    if has_qutip:
        qutip_sv = (ket('000') - ket('011') - ket('110') - ket('111')).unit()
        assert converter.to_perceval(qutip_sv) == expected_sv


@pytest.mark.skipif(not has_qiskit, reason="Requires qiskit")
def test_statevector_converter_failures():
    converter = StatevectorConverter(encoding=Encoding.DUAL_RAIL)

    p_sv = StateVector('|{P:V},{P:V},{P:V}>') - StateVector('|{P:H},{P:V},{P:V}>')  # Encoding mismatch
    p_sv2 = StateVector('|0,1,0,1,0>')  # Missing a mode to be dual rail

    with pytest.raises(ValueError):
        converter.to_qiskit(p_sv)

    with pytest.raises(ValueError):
        converter.to_qiskit(p_sv2)
