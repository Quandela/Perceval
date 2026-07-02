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

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Callable

from .async_getter import AsyncGetter
from .error_mitigation import AbstractMitigation
from .computation import Computation
from .computation_iterator import ComputationIterator
from .platform_specs import PlatformSpecs
from .check_cancel import call_and_check_cancel
from .command import Command

from perceval.utils import ProgressCallback, partial_progress_callable, ContextManager, NoiseModel, PMetadata
from perceval.utils.constants import KEY_RESULTS


class AbstractComputer(ABC):

    EMT_POST_PROGRESS_START = 0.8

    def __init__(self):
        self._commands: dict[str, Command] = {}
        self._error_mitigations: list[AbstractMitigation] | None = None
        self._parameters: dict[str, Any] = {}
        self.reset_parameters()

    def _register_command(self, command: Command) -> None:
        """
        :param command: A Command to add to the possible commands of this Computer.
            The associated method must be able to handle **kwargs
        """
        self._commands[command.name] = command

    def get_command(self, command_name: str) -> Command:
        if command_name not in self._commands:
            raise ValueError(f"Command '{command_name}' doesn't exist in {self.__class__.__name__}")
        return self._commands[command_name]

    @property
    def mitigations(self) -> list[AbstractMitigation] | None:
        return self._error_mitigations

    @mitigations.setter
    def mitigations(self, error_mitigations: list[AbstractMitigation] | None):
        self._error_mitigations = error_mitigations

    def _get_local_mitigations(self) -> list[AbstractMitigation]:
        # Internal use: defines which mitigations to apply locally
        return self._error_mitigations or []

    @property
    def available_commands(self) -> list[str]:
        return list(self._commands)  # Makes a copy

    def reset_parameters(self):
        # May be overloaded to have default parameters
        self._parameters.clear()

    @property
    def parameters(self) -> dict[str, Any]:
        return self._parameters

    @parameters.setter
    def parameters(self, parameters: dict[str, Any]):
        self._parameters = parameters

    @property
    def available_parameters(self) -> dict[str, str]:
        """
        :return: A dictionary describing all the available parameters keys and their meaning.
        """
        return {}

    def validate_single(self, computation: Computation) -> None:
        """
        :param computation: The computation to validate. This is a computation that is the result of mitigation decomposition.
        :return:
        """
        if computation.command.name not in self._commands:
            raise ValueError(f"Command '{computation.command.name}' doesn't exist in {self.__class__.__name__}")

    def _handle_iterator(self, comp: Computation | ComputationIterator, out: dict | None)\
            -> tuple[dict, Callable[[dict], None]]:
        """
        :param comp: A Computation or an iterator
        :param out: A potentially externally given dict to insert the results in. If None, it is instantiated here
        :return: out if given or an empty new dict. Also returns a method to be used on the results of the computation to add it to out.
        """
        if out is None:
            out = dict()

        if isinstance(comp, ComputationIterator):
            return out, comp.make_inserter(out)

        return out, lambda res: out.update(res)

    def extend_computation_keep_original(self, computation: Computation | ComputationIterator) -> list[tuple[list[Computation], Computation]]:
        computations = []
        for comp in computation:
            computations.append((self._extend_computation(comp), comp))
        return computations

    def extend_computation(self, computation: Computation | ComputationIterator) -> list[list[Computation]]:
        computations = []
        for comp in computation:
            computations.append(self._extend_computation(comp))
        return computations

    def _extend_computation(self, comp: Computation) -> list[Computation]:
        """
        :param comp: The computation to be executed as the final step
        :return: The list of all computations to execute
        """
        if comp.command.apply_emt:
            return self._prepare_sub_computations([comp], self._get_local_mitigations())
        return [comp]

    def _prepare_sub_computations(self, computations: list[Computation], emts: list[AbstractMitigation]) -> list[Computation]:
        if len(emts) == 0:
            return computations

        res = []
        for computation in computations:
            res += self._prepare_sub_computations(emts[0].extend_computation(computation, self.noise), emts[1:])

        return res

    def post_process(self,
                     original_computation: Computation,
                     results: list[dict | AsyncGetter],
                     noise: NoiseModel,
                     emts: list[AbstractMitigation] = None,
                     progress_callback: ProgressCallback = None) -> dict:
        if not original_computation.command.apply_emt:
            emts = None
        return self._post_process(original_computation, emts or [], results, noise, progress_callback)[0]

    def _post_process(self, computation: Computation, emts: list[AbstractMitigation], results: list,
                      noise: NoiseModel,
                      progress_callback: ProgressCallback = None, current_index: int = 0) -> tuple[dict, int]:
        # current_index supposes that results are in the order requested by self.extend_computation()
        if len(emts) == 0:
            # Do we split this evenly for all mitigations ?
            if progress_callback is not None:
                progress_callback((current_index + 1) / len(results), "Post processing results")
            res = results[current_index]
            if isinstance(res, AsyncGetter):
                return results[current_index].get_results(), current_index + 1
            else:
                return results[current_index], current_index + 1

        computations = emts[0].extend_computation(computation, noise)
        res: list[dict] = []
        for comp in computations:
            sub_res, current_index = self._post_process(comp, emts[1:], results, noise, progress_callback, current_index)
            res.append(sub_res)

        return emts[0].parse_results(computation, res, noise), current_index

    def execute(self, computation: Computation | ComputationIterator, out: dict = None, progress_callback: ProgressCallback = None) -> dict:
        """Synchronous execution of computation

        :param computation: A Computation or an iterator to execute.
        :param out: A potentially externally given dict where to place the results.
         If the computation is an iterator, it can receive be used to retrieve partial results.
        :param progress_callback: A ProgressCallback to monitor the progress and potentially cancel the execution.
        """
        res, inserter = self._handle_iterator(computation, out)

        try:
            computation.validate()
            computations = self.extend_computation_keep_original(computation)
            self._execute_all(computations, inserter, progress_callback)
        except Exception as e:
            inserter({KEY_RESULTS: f"{type(e).__name__}: {e}"})
            raise

        if progress_callback is not None:
            progress_callback(1., "Finished!")

        return res

    def _execute_all(self,
                     computations: list[tuple[list[Computation], Computation]],
                     inserter: Callable[[dict], None],
                     progress_callback: ProgressCallback = None) -> None:
        # This method may be overloaded by some executors to be able to "factorize" some computations
        # For example by factorizing a chip voltage appliance or compilation
        n_iter = len(computations)

        for i, (comp_list, original_computation) in enumerate(computations):
            batch_callback = partial_progress_callable(progress_callback, i / n_iter, (i + 1) / n_iter)
            comp_callback = partial_progress_callable(batch_callback, max_val=self.EMT_POST_PROGRESS_START)
            n = len(comp_list)
            res = []
            # Step 1: we get the results for every sub-computation
            for j, comp in enumerate(comp_list):
                sub_comp_callback = partial_progress_callable(comp_callback, j / n, (j + 1) / n)
                res.append(self._execute_single(comp, sub_comp_callback))
                if call_and_check_cancel(sub_comp_callback, 1., "Sub-computation complete"):
                    return

            # Step 2: we post-process for the current computation and insert it in the results
            inserter(self.post_process(original_computation, res, self.noise, self._error_mitigations,
                                       partial_progress_callable(batch_callback, self.EMT_POST_PROGRESS_START)))

            if len(computations) > 1:
                progress_msg = "All iterations complete" if i == n_iter - 1 else "Switching to next iteration"
                if call_and_check_cancel(batch_callback, 1, progress_msg):
                    return

    def _execute_single(self, computation: Computation, progress_callback: ProgressCallback = None) -> dict:
        # Most of the AbstractComputer specific implementation is in the self._commands
        self.validate_single(computation)
        with self._reserve_resource():
            return self._execute_command(computation, progress_callback)

    @abstractmethod
    def _execute_command(self, computation: Computation, progress_callback: ProgressCallback = None) -> dict:
        pass

    def execute_async(self, computation: Computation | ComputationIterator) -> tuple[list[AbstractMitigation] | None, NoiseModel, list[list[AsyncGetter]]]:
        """
        Asynchronous execution of computation

        :param computation: The computation to execute
        :return: The Error mitigations with which the computation is executed, and the list of objects that can be used to get the results
        """
        computation.validate()
        computations = self.extend_computation(computation)
        return deepcopy(self._get_local_mitigations()), deepcopy(self.noise), self._execute_all_async(computations)

    def get_results(self, computation: Computation | ComputationIterator,
                    mitigations: list[AbstractMitigation],
                    noise: NoiseModel,
                    async_getters: list[list[AsyncGetter]],
                    out: dict = None) -> dict[str, Any]:
        """
        Get the results for an asynchronous computation
        :param computation: The original computation that was executed
        :param mitigations: The list of mitigations that were applied when the computation has been launched (as returned by execute_async)
        :param noise: The noise model with which the computations were executed
        :param async_getters: The list of async_getters that point to the executions of the computation (as returned by execute_async)
        :param out: A dictionary where to place the results.
        """
        res, inserter = self._handle_iterator(computation, out)

        try:
            for getters, comp in zip(async_getters, computation):
                inserter(self.post_process(comp, getters, noise, mitigations))
        except Exception as e:
            inserter({KEY_RESULTS: str(e)})
            raise

        return res

    def _execute_all_async(self, computations: list[list[Computation]]) -> list[list[AsyncGetter]]:
        """
        :return: The list of AsyncGetter that can be used to get the results
        """
        return [[self._execute_single_async(comp) for comp in computation] for computation in computations]

    def _execute_single_async(self, computation: Computation) -> AsyncGetter:
        """
        :param computation: The computation to execute
        :return: An AsyncGetter that can be used to get the results for this computation
        """
        # For local: make a thread, launch it and return it
        # For remote: send to the cloud and return the id - May wait for availability
        self.validate_single(computation)
        return self._execute_command_async(computation)

    @abstractmethod
    def _execute_command_async(self, computation: Computation) -> AsyncGetter:
        pass

    @property
    @abstractmethod
    def is_remote(self) -> bool:
        pass

    def acquire(self) -> ContextManager:
        """
        This method can be used to set up the AbstractComputer before the use of the :code:`execute` method.
        For example, it can be overloaded to warm up internal tools, or empty some cache at the end of a computation.
        """
        return ContextManager()

    def _reserve_resource(self) -> ContextManager:
        """
        This method is used internally when computing basic computations (after error mitigation extension).

        It can be overloaded to prevent the resources of this AbstractComputer to be used more than once at the same time,
        by waiting for the release of its resources.
        """
        return ContextManager()

    @property
    def specs(self) -> PlatformSpecs:
        specs = PlatformSpecs()
        specs.commands = list(self._commands.values())
        if self.available_parameters:
            specs.parameters = self.available_parameters
        specs.type = self.type
        specs.pcvl_version = PMetadata.version()
        return specs

    @property
    def noise(self):
        return NoiseModel()

    @noise.setter
    @abstractmethod
    def noise(self, noise: NoiseModel):
        pass

    @property
    @abstractmethod
    def performance(self):
        pass

    @property
    @abstractmethod
    def type(self):
        pass

    def apply_configuration(self,
                            mitigations: list[AbstractMitigation] = None,
                            noise: NoiseModel = None,
                            parameters: dict[str, Any] = None) -> ContextManager:
        """
        :param mitigations: The mitigations to apply within the ContextManager. If None, nothing is changed
        :param noise: The noise model to apply within the ContextManager. If None, nothing is changed
        :param parameters: The parameters to apply within the ContextManager. If None, nothing is changed
        :return: A ContextManager that applies the given arguments to the computer (noise, mitigations, parameters)
          at enter and reset the parameters to the previous values at exit
        """
        starting_mitigations = self.mitigations
        starting_noise = self.noise if noise is not None else None
        starting_parameters = self.parameters if parameters is not None else None

        def apply(mitigations_: list[AbstractMitigation] | None, noise_: NoiseModel | None, parameters_: dict[str, Any] | None, force = False):
            if mitigations_ is not None or force:
                self.mitigations = mitigations_
            if noise_ is not None:
                self.noise = noise_
            if parameters_ is not None:
                self.parameters = parameters_

        return ContextManager(lambda: apply(mitigations, noise, parameters),
                              lambda: apply(starting_mitigations, starting_noise, starting_parameters, force = True))
