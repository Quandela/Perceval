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
from __future__ import annotations

from collections.abc import Callable

import exqalibur as xq
from perceval.components import ACircuit, Circuit, GenericInterferometer, BS, PS, catalog
from perceval.utils import Matrix, P
from perceval.utils.logging import get_logger, channel
from perceval.serialization import serialize_binary, deserialize_circuit


class CircuitOptimizer:
    """
    CircuitOptimizer is a tool designed to set up a circuit with enough variable parameters
    (i.e. sufficient degrees of freedom) so that it reproduces to the behavior of any other unitary circuit/matrix.

    Be aware that some circuits are not "universal interferometers" and thus cannot reach every arbitrary unitary
    matrix. However, the optimizer will get as close as possible. The metric to measure the distance between the target
    unitary and the optimized interferometer is the matrix fidelity available in `perceval.utils.algorithms.norm`.

    CircuitOptimizer can be configured with the following parameters:

    :param threshold: Error threshold = 1-fidelity - i.e. the lower the threshold, the better the output fidelity
                      (default 1e-6)
    :param ntrials: Number of optimization trials (default 4)
    :param max_eval_per_trial: maximum number of evaluations per optimization trial (default 200000)
    """

    def __init__(self, threshold: float = 1e-6, ntrials: int = 4, max_eval_per_trial: int = 200000):

        self.threshold = threshold
        self.trials = ntrials
        self.max_eval_per_trial = max_eval_per_trial

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

    @property
    def max_eval_per_trial(self):
        return self._max_eval_per_trial

    @max_eval_per_trial.setter
    def max_eval_per_trial(self, value):
        self._max_eval_per_trial = value

    def optimize(self,
                 target: ACircuit | Matrix,
                 template: ACircuit,
                 empty_mode_list: list[int] = None
                 ) -> tuple[ACircuit, float]:
        """
        Optimize a template circuit unitary's fidelity with a target matrix or circuit.

        :param target: The target unitary circuit or matrix
        :param template: A circuit with variable parameters (supports only beam splitters and phase shifters)
        :param empty_mode_list: list of the modes without input photon,
                                which are ignored during optimisation as this does not alter the results (default [])
        :return: A tuple of the best optimized circuit and its fidelity to the target

        >>> def mzi(i):
        >>>    return Circuit(2) // PS(P(f"phi_1_{i}")) // BS() // PS(P(f"phi_2_{i}")) // BS()
        >>> def ps(i):
        >>>    return PS(P(f"phi_3_{i}"))
        >>> template = GenericInterferometer(12, mzi, phase_shifter_fun_gen=ps, phase_at_output=True)
        >>> random_unitary = Matrix.random_unitary(12)
        >>> result_circuit, fidelity = CircuitOptimizer().optimize(random_unitary, template)
        """
        if isinstance(target, ACircuit):
            target = target.compute_unitary()

        if template.m != target.shape[0]:
            raise ValueError(f"Template circuit and target size should be the same ({template.m} != {target.shape[0]})")

        if target.is_symbolic():
            raise TypeError("Target must be numeric")

        if empty_mode_list is None:
            empty_mode_list = []

        optimizer = xq.CircuitOptimizer(target, serialize_binary(template), empty_mode_list)
        optimizer.set_max_eval_per_trial(self._max_eval_per_trial)
        optimizer.set_threshold(self._threshold)
        optimized_circuit = deserialize_circuit(optimizer.optimize(self._trials))
        return optimized_circuit, optimizer.fidelity

    def optimize_rectangle(self,
                           target: Matrix,
                           template_component_generator_func: Callable[[int], ACircuit] = None,
                           phase_at_output: bool = True,
                           allow_error: bool = False,
                           empty_mode_list: list[int] = None
                           ) -> ACircuit:
        """
        Optimize a rectangular circuit to reach a target unitary matrix fidelity.

        :param target: Target unitary matrix
        :param template_component_generator_func: Function which generates a base component of the output rectangular
                                                  circuit (signature(int) -> ACircuit)
                                                  (default generates a MZI with two variable phases)
        :param phase_at_output: If True, a layer of phase shifters is added at the output of the circuit.
                                Otherwise, the layer is at the input (default True)
        :param allow_error: If True, this call will not raise an error when the best fidelity is below threshold
                            Otherwise, raises an error (default False)
        :param empty_mode_list: list of the modes without input photon,
                                which are ignored during optimisation as this does not alter the results (default [])
        """
        def _gen_ps(i: int):
            return PS(P(f"phL_{i}"))

        if template_component_generator_func is None:
            mzi_name = "mzi phase first"
            get_logger().debug(f"Using default MZI ({mzi_name}) for rectangular optimization", channel.general)
            template_component_generator_func = catalog[mzi_name].generate
        template = GenericInterferometer(
            target.shape[0],
            template_component_generator_func,
            phase_shifter_fun_gen=_gen_ps,
            phase_at_output=phase_at_output)
        result_circuit, fidelity = self.optimize(target, template, empty_mode_list)
        if fidelity < 1 - self._threshold:
            if allow_error:
                get_logger().warn(f"Optimization converged with poor fidelity ({fidelity})", channel.general)
            else:
                raise RuntimeError(f"Optimization did not converge to expected threshold ({self._threshold})")
        return result_circuit
