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

from perceval.components import GenericInterferometer, catalog, PS
from perceval.backends import SLOSBackend
from perceval.utils import BasicState, P

import numpy as np
import pytest


size_identity = 6
mzi_generator_func = catalog['mzi phase last'].generate


@pytest.mark.parametrize('interferometer', [
    GenericInterferometer(size_identity, mzi_generator_func),
    GenericInterferometer(size_identity, mzi_generator_func,
                          phase_shifter_fun_gen=lambda i: PS(P(f"phi{i}")), phase_at_output=True),
    GenericInterferometer(size_identity, mzi_generator_func,
                          phase_shifter_fun_gen=lambda i: PS(P(f"phi{i}")), phase_at_output=False)
    ])
def test_set_identity(interferometer):
    input_state = BasicState([1]*size_identity)  # One photon per mode as input
    interferometer.set_identity_mode()
    slos = SLOSBackend()
    slos.set_circuit(interferometer)
    slos.set_input_state(input_state)
    assert slos.probability(output_state=input_state) == pytest.approx(1)  # Detect one photon per mode


def test_set_param_list():
    size = 12
    values = [.0, .1, .2, .3, .4, .5, .6, .7, .8, .9]

    interferometer = GenericInterferometer(size, mzi_generator_func)
    interferometer.set_param_list(values, (0, 0), m=2)
    # With m=2, online one row of phase shifters get impacted (on mode 1, given the mzi we used)
    # The 10 first phase shifter of this row get phi=values[idx]
    for idx, phase_shifter_pos in enumerate([1, 3, 7, 9, 13, 15, 19, 21, 25, 27]):
        assert float(interferometer[1, phase_shifter_pos].get_parameters()[0]) == pytest.approx(values[idx])
    # next phase shifters still have a variable phi:
    assert interferometer[1, 31].get_parameters()[0].defined == False
    params_with_numerical_value = [p for p in interferometer.get_parameters() if p.defined]
    assert len(params_with_numerical_value) == len(values)

    # Moving 2 MZI down, means 4 modes down
    interferometer = GenericInterferometer(size, mzi_generator_func)
    interferometer.set_param_list(values, (0, 2), m=2)
    for idx, phase_shifter_pos in enumerate([1, 3, 7, 9, 13, 15, 19, 21, 25, 27]):
        assert float(interferometer[5, phase_shifter_pos].get_parameters()[0]) == pytest.approx(values[idx])

    # Moving 1 MZI right, means 6 components right
    interferometer = GenericInterferometer(size, mzi_generator_func)
    interferometer.set_param_list(values, (1, 0), m=2)
    for idx, phase_shifter_pos in enumerate([1, 3, 7, 9, 13, 15, 19, 21, 25, 27]):
        assert float(interferometer[1, phase_shifter_pos+6].get_parameters()[0]) == pytest.approx(values[idx])

    # Starting too right, can get out of the interferometer
    interferometer = GenericInterferometer(size, mzi_generator_func)
    with pytest.raises(ValueError):
        interferometer.set_param_list(values, (3, 0), m=2)

    # Reshaping by giving a higher m value, will impact more modes on instertion
    interferometer = GenericInterferometer(size, mzi_generator_func)
    interferometer.set_param_list(values, (0, 0), m=4)
    for idx, (x, y) in enumerate([(1,1), (1,3), (3,1), (3,3), (2,3), (2,5), (1,7), (1,9), (3,7), (3,9)]):
        assert float(interferometer[x, y].get_parameters()[0]) == pytest.approx(values[idx])


def test_set_params_from_other():
    small_interferometer = GenericInterferometer(4, mzi_generator_func)
    small_interferometer.set_identity_mode()  # Give a value to phases

    big_interferometer = GenericInterferometer(8, mzi_generator_func,
                                    phase_shifter_fun_gen=lambda i: PS(P(f"phi_L{i}")))
    big_interferometer.set_params_from_other(small_interferometer, (0, 0))
    big_interferometer.remove_phase_layer()

    for (x, y) in [(1,1), (1,3), (3,1), (3,3), (2,3), (2,5), (1,7), (1,9), (3,7), (3,9)]:
        assert float(big_interferometer[x, y].get_parameters()[0]) == pytest.approx(np.pi)
    params_with_numerical_value = [p for p in big_interferometer.get_parameters() if p.defined]
    assert len(params_with_numerical_value) == len(small_interferometer.get_parameters())

    with pytest.raises(ValueError):
        big_interferometer.set_params_from_other(small_interferometer, (6, 2))
