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

from perceval.components import Experiment
from perceval.utils.logging import get_logger, channel
from perceval.utils.constants import KEY_MAX_SHOTS

from .abstract_computer import AComputer
from .computation import Computation
from .computation_iterator import ComputationIterator
from .execution import Execution


class ExecutionFactory:
    """Build computations and executions for a computer and an experiment.

    The factory exposes the standard ``probs``, ``samples``, and ``sample_count`` commands as properties.
    Accessing one of these properties creates a new :class:`Execution`.
    Commands registered by a custom computer are exposed dynamically in the same way
    (but will not get autocompletion from your IDE).

    Iterations added to the factory are included in every subsequently built computation until
    :meth:`clear_iterations()` is called.

    :param computer: Computer that will perform the executions.
    :param experiment: Experiment used by the computations.
    :param max_shots_per_call: Default maximum number of shots for each built computation.
        A positive value is mandatory for remote computers.
    """

    def __init__(self, computer: AComputer, experiment: Experiment, max_shots_per_call: int = None):
        self.computer = computer
        self.experiment = experiment
        self.max_shots_per_call = max_shots_per_call
        self._iterator = []

        self.default_job_name = None

        if self.max_shots_per_call:
            self.max_shots_per_call = int(self.max_shots_per_call)
            if self.max_shots_per_call < 1:
                raise RuntimeError('`max_shots_per_call` must be a positive value')
        # max_shots_per_call must be found in **kwargs when the computer is remote.
        # This condition is forced because the user will consume credits on the cloud and needs to set an upper bound
        if computer.is_remote and not self.max_shots_per_call:
            raise RuntimeError('Please input a `max_shots_per_call` value when using a RemoteComputer')

    def add_iteration(self, **kwargs):
        """Add an iteration to subsequently built computations.

        Iteration parameters are validated when :meth:`build_computation` creates the
        :class:`ComputationIterator`, or an :class:`Execution` is created through a property.

        :param kwargs: List of accepted keywords:

           - ``circuit_params``: numerical values keyed by circuit parameter name
           - ``input_state``: :class:`BasicState`
           - ``min_detected_photons``: minimum accepted photon count
           - ``max_samples``: maximum number of samples to collect
           - ``max_shots``: maximum number of shots to perform
           - ``postselect``: :class:`PostSelect` condition
        """
        get_logger().info("Add 1 iteration to ExecutionFactory", channel.general)
        self._iterator.append(kwargs)

    def add_iteration_list(self, iterations: list[dict]):
        """Add several iterations to subsequently built computations.

        The dictionaries are appended in order and follow the same format as :meth:`add_iteration`.

        :param iterations: Ordered iteration parameter dictionaries.
        """
        get_logger().info(f"Add {len(iterations)} iterations to ExecutionFactory", channel.general)
        for iter_params in iterations:
            self._iterator.append(iter_params)

    def clear_iterations(self):
        """Remove all iterations currently stored by the factory."""
        get_logger().info("Clear all iterations in ExecutionFactory", channel.general)
        self._iterator.clear()

    @property
    def n_iterations(self):
        """Return the number of iterations currently stored by the factory."""
        return len(self._iterator)

    def build_computation(self, name: str) -> Computation | ComputationIterator:
        """Build a computation for a command supported by the computer.

        If the factory contains iterations, the result is a :class:`ComputationIterator` wrapping
        the base computation. Otherwise, a :class:`Computation` is returned. The factory's
        ``max_shots_per_call`` value is added as the base ``max_shots`` parameter when configured.

        :param name: Name of a command registered by the computer.
        :return: A computation, or a computation iterator when iterations have been added.
        :raises ValueError: If the computer does not support ``name``.
        """
        command = self.computer.get_command(name)
        comp = Computation(command, self.experiment)
        if self.max_shots_per_call is not None and any(sig[0] == KEY_MAX_SHOTS for sig in command.signature):
            comp.add_params(**{KEY_MAX_SHOTS: self.max_shots_per_call})

        if self.n_iterations > 0:
            comp = ComputationIterator(comp)
            for it in self._iterator:
                comp.add_iteration(**it)

        return comp

    def build_execution(self, computation: Computation | ComputationIterator) -> Execution:
        """Build a new execution for this factory's computer.

        If :attr:`default_job_name` is set, that name is assigned to the new execution.

        :param computation: Computation or iterator to execute.
        :return: A new, unsent execution.
        """
        execution = Execution(computation, self.computer)

        if self.default_job_name is not None:
            execution.name = self.default_job_name

        return execution

    @property
    def probs(self):
        """Build a new execution for the standard ``probs`` command."""
        return self.build_execution(self.build_computation('probs'))

    @property
    def samples(self):
        """Build a new execution for the standard ``samples`` command."""
        return self.build_execution(self.build_computation('samples'))

    @property
    def sample_count(self):
        """Build a new execution for the standard ``sample_count`` command."""
        return self.build_execution(self.build_computation('sample_count'))

    def __getattr__(self, item):
        """Build an execution for a custom command registered by the computer."""
        return self.build_execution(self.build_computation(item))
