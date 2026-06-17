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
import random
from copy import deepcopy

import numpy as np
from packaging.version import Version

from perceval.components.compiled_circuit import CompiledCircuit, CompiledCircuitVersion
from perceval.components.core_catalog.mzi import MZIPhaseFirst
from perceval.components.experiment import Experiment

def test_inheritance():
    circuit = CompiledCircuit("chip name", 2, [], CompiledCircuitVersion(Version("1.0")))
    exp = Experiment(3, None, "-")
    exp.add(1, circuit)

    exp = Experiment(circuit, None, "-")

def test_compute_unitary():
    circuit = CompiledCircuit("chip name", MZIPhaseFirst().build_circuit(), [0., 1.], CompiledCircuitVersion(Version("1.0")))
    assert np.allclose(circuit.compute_unitary(), MZIPhaseFirst().build_circuit(phi_a = 0, phi_b = 1.).compute_unitary())

def test_phase_noise():
    circuit = CompiledCircuit("chip name", 2, [0., 1.], CompiledCircuitVersion(Version("1.0")))

    rng = random.Random(42)

    circuit_cp = deepcopy(circuit)
    circuit_cp.apply_phase_noise(0.1, 0, rng)

    for original, phase in zip(circuit.parameters, circuit_cp.parameters):
        assert - 0.1 <= phase - original <= 0.1

    rng = random.Random(42)
    circuit_cp2 = deepcopy(circuit)
    circuit_cp2.apply_phase_noise(0.1, 0, rng)

    assert np.allclose(circuit_cp.parameters, circuit_cp2.parameters)
    assert not np.allclose(circuit.parameters, circuit_cp.parameters)
