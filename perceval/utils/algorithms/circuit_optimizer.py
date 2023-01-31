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

from typing import Callable, Tuple, Union

import exqalibur as xq
from perceval.components import ACircuit, Circuit, PS
from perceval.utils import Matrix, P
from perceval.serialization import serialize_binary, deserialize_circuit


class CircuitOptimizer:

    def __init__(self):
        self._threshold = 1e-6
        self._trials = 4

    @property
    def threshold(self):
        return self._threshold

    @threshold.setter
    def threshold(self, value):
        if not (0 < value < 1):
            raise ValueError("Fitness threshold should be between 0 and 1")
        self._threshold = value

    @property
    def trials(self):
        return self._trials

    @trials.setter
    def trials(self, value):
        self._trials = value

    def optimize(self,
                 target: Union[ACircuit, Matrix],
                 template: ACircuit
                 ) -> Tuple[float, ACircuit]:
        if isinstance(target, ACircuit):
            target = target.compute_unitary()
        assert template.m == target.shape[0], \
            f"Template circuit and target size should be the same ({template.m} != {target.m})"
        assert not target.is_symbolic(), "Target must not contain variables"

        optimizer = xq.CircuitOptimizer(serialize_binary(target), serialize_binary(template))
        optimizer.set_threshold(self._threshold)
        optimized_circuit = deserialize_circuit(optimizer.optimize(self._trials))
        return optimizer.fidelity, optimized_circuit

    def optimize_rectangle(self,
                           target: Matrix,
                           template_component_generator_func: Callable[[int],ACircuit],
                           phase_at_output: bool = False,
                           allow_error: bool = False,
                           ):
        def _gen_ps(i: int):
            return PS(P(f"phL_{i}"))

        template = Circuit.generic_interferometer(
            target.shape[0],
            template_component_generator_func,
            phase_shifter_fun_gen=_gen_ps,
            phase_at_output=phase_at_output)
        fidelity, result_circuit = self.optimize(target, template)
        if not allow_error and fidelity < 1 - self._threshold:
            raise RuntimeError(f"Optimization did not convergence to expected threshold ({self._threshold})")
        return result_circuit
