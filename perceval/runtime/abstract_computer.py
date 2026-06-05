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
from typing import Any, TypeVar

from .error_mitigation import AbstractMitigation
from .computation import Computation
from .computation_iterator import ComputationIterator
from .platform_specs import PlatformSpecs

from perceval.utils import ProgressCallback, partial_progress_callable, ContextManager, NoiseModel

AsyncGetter = TypeVar("AsyncGetter")  # Make abstract class ?


class AbstractComputer(ABC):

    EMT_POST_PROGRESS_START = 0.8

    def __init__(self):
        self._commands = []  # Commands can be added after this __init__ by subclasses
        # TODO: link to method to make command parameters checking automatic ?
        self._error_mitigations: list[AbstractMitigation] = []
        self._parameters: dict[str, Any] = {}

    def set_mitigations(self, error_mitigations: list[AbstractMitigation]):
        # TODO: The interface to set the mitigation is still to be defined
        self._error_mitigations = error_mitigations

    @property
    def available_commands(self) -> list[str]:
        return list(self._commands)  # Makes a copy

    def set_parameters(self, parameters: dict[str, Any]):
        for parameter, value in parameters.items():
            assert parameter in self.available_parameters, f"Unknown parameter '{parameter}'"
            self._parameters[parameter] = value

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

    def _handle_iterator(self, comp: Computation | ComputationIterator, emts: list[AbstractMitigation] = None) -> tuple[Computation, list[AbstractMitigation | ComputationIterator]]:
        if comp.command.apply_emt:
            emts = emts if emts is not None else self._error_mitigations
        else:
            emts = []
        if isinstance(comp, ComputationIterator):
            return comp.base_computation, [comp] + emts

        return comp, emts

    def extend_computation(self, comp: Computation | ComputationIterator) -> list[Computation]:
        """
        :param comp: The computation to be executed as the final step
        :return: The list of all computations to execute
        """
        computation, emts = self._handle_iterator(comp)
        return self._prepare_sub_computations([computation], emts)

    def _prepare_sub_computations(self, computations: list[Computation], emts: list[AbstractMitigation | ComputationIterator]) -> list[Computation]:
        if len(emts) == 0:
            return computations

        res = []
        for computation in computations:
            res += self._prepare_sub_computations(emts[0].extend_computation(computation, self.noise), emts[1:])

        return res

    def post_process(self,
                     original_computation: Computation | ComputationIterator,
                     results: list[dict],
                     progress_cb: ProgressCallback = None):
        computation, emts = self._handle_iterator(original_computation)
        return self._post_process(computation, emts, results, self.noise, True, progress_cb)[0]

    def _post_process(self, computation: Computation, emts: list[AbstractMitigation | ComputationIterator], results: list,
                      noise: NoiseModel,
                      is_sync: bool, progress_cb: ProgressCallback = None, current_index: int = 0) -> tuple[dict, int]:
        # current_index supposes that results are in the order requested by self.extend_computation()
        if len(emts) == 0:
            # Do we split this evenly for all mitigations ?
            if progress_cb is not None:
                progress_cb((current_index + 1) / len(results), "Post processing results")
            if is_sync:
                return results[current_index], current_index + 1
            else:
                return self._load_async_result(results[current_index]), current_index + 1

        computations = emts[0].extend_computation(computation, noise)
        res: list[dict] = []
        for comp in computations:
            sub_res, current_index = self._post_process(comp, emts[1:], results, noise, is_sync, progress_cb, current_index)
            res.append(sub_res)

        return emts[0].parse_results(computation, res, noise), current_index

    def execute(self, computation: Computation | ComputationIterator, progress_cb: ProgressCallback = None) -> Any:
        """Synchronous execution of computation"""
        computation.validate()
        computations = self.extend_computation(computation)

        all_results = self._execute_all(computations, partial_progress_callable(progress_cb, max_val=self.EMT_POST_PROGRESS_START))
        return self.post_process(computation, all_results, partial_progress_callable(progress_cb, self.EMT_POST_PROGRESS_START))

    def _execute_all(self, computations: list[Computation], progress_cb: ProgressCallback = None) -> list[dict]:
        # This method may be overloaded by some executors to be able to "factorize" some computations
        # For example by factorizing a chip voltage appliance or compilation
        n = len(computations)
        return [self._execute_single(comp, partial_progress_callable(progress_cb, i / n, (i + 1) / n))
                for i, comp in enumerate(computations)]

    def _execute_single(self, computation: Computation, progress_cb: ProgressCallback = None) -> dict:
        # Most of the AbstractComputer specific implementation is in the self._commands
        self.validate_single(computation)
        with self._reserve_resource():
            return self._execute_command(computation, progress_cb)

    @abstractmethod
    def _execute_command(self, computation: Computation, progress_cb: ProgressCallback = None) -> dict:
        pass

    def execute_async(self, computation: Computation | ComputationIterator) -> tuple[list[AbstractMitigation], NoiseModel, list[AsyncGetter]]:
        """
        Asynchronous execution of computation

        :param computation: The computation to execute
        :return: The Error mitigations with which the computation is executed, and the list of objects that can be used to get the results
        """
        computation.validate()
        computations = self.extend_computation(computation)

        return deepcopy(self._error_mitigations), deepcopy(self.noise), self._execute_all_async(computations)

    def get_results(self, computation: Computation, mitigations: list[AbstractMitigation], noise: NoiseModel, async_getters: list[AsyncGetter]) -> dict[str, Any]:
        """
        Get the results for an asynchronous computation
        :param computation: The original computation that was executed
        :param mitigations: The list of mitigations that were applied when the computation has been launched (as returned by execute_async)
        :param noise: The noise model with which the computations were executed
        :param async_getters: The list of async_getters that point to the executions of the computation (as returned by execute_async)
        """
        computation, emts = self._handle_iterator(computation, mitigations)
        return self._post_process(computation, emts, async_getters, noise, False)[0]

    def _execute_all_async(self, computations: list[Computation]) -> list[AsyncGetter]:
        """
        :return: The list of AsyncGetter that can be used to get the results
        """
        return [self._execute_single_async(comp) for comp in computations]

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

    @abstractmethod
    def _load_async_result(self, async_getter: AsyncGetter) -> dict:
        """
        :param async_getter: The object describing where to get the result of a computation
        :return: The results of the computation
        """

    @abstractmethod
    def is_complete(self, async_getter: AsyncGetter) -> bool:
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
        return PlatformSpecs()

    @property
    def noise(self):
        return NoiseModel()

    @noise.setter
    @abstractmethod
    def noise(self, noise):
        pass

    @property
    @abstractmethod
    def performance(self):
        pass

    @property
    @abstractmethod
    def type(self):
        pass
