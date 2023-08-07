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

import perceval as pcvl
from perceval.components.unitary_components import BS, PS, Unitary
import numpy as np

def get_interferometer(n):
    def _gen_mzi(i: int):
        return pcvl.catalog["mzi phase last"].build_circuit(theta_a=0.42, theta_b=0.42,
                                                            phi_a=np.pi+i*0.1, phi_b=np.pi/2)
    return pcvl.GenericInterferometer(n, _gen_mzi)


def simulate_sampling(shots, circuit, input_state):
    clifford = pcvl.Clifford2017Backend()
    clifford.set_circuit(circuit)
    clifford.set_input_state(input_state)
    for i in range(shots):
        clifford.sample()


def test_bosonsampling_clifford_6(benchmark):
    benchmark(simulate_sampling, shots=100,
              circuit=get_interferometer(6), input_state=pcvl.BasicState([1] * 6))


def test_bosonsampling_clifford_8(benchmark):
    benchmark(simulate_sampling, shots=20,
              circuit=get_interferometer(8), input_state=pcvl.BasicState([1] * 8))
