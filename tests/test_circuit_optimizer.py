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
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from perceval.utils.algorithms.circuit_optimizer import CircuitOptimizer
from perceval.utils.algorithms import norm
from perceval.components import BS, PS, Circuit
from perceval.utils import P, Matrix

import pytest


def test_circuit_optimizer():
    perfect_R = 0.5

    for size in range(4, 17):
        circuit_optimizer = CircuitOptimizer()
        def mzi(i):
            return Circuit(2) // PS(P(f"phi_1_{i}")) // BS(BS.r_to_theta(perfect_R)) \
                // PS(P(f"phi_2_{i}")) // BS(BS.r_to_theta(perfect_R))
        def ps(i):
            return PS(P(f"phi_3_{i}"))
        template_interferometer = Circuit.generic_interferometer(size, mzi, phase_shifter_fun_gen=ps, phase_at_output=True)

        random_unitary = Matrix.random_unitary(size)
        fidelity, result_circuit = circuit_optimizer.optimize(random_unitary, template_interferometer)
        assert 1 - fidelity < circuit_optimizer.threshold
        assert norm.fidelity(result_circuit.compute_unitary(), random_unitary) == pytest.approx(fidelity)
