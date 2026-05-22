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

from typing import Generator

from perceval.runtime.command import Command
from perceval.runtime.computation import Computation


# just an adaptation of the perceval class; Must follow some of the EMT interface and some of the Computation interface
# TODO: merge this with the ParameterIterator class
class Iterator:
    # This class is only a user-friendly way to create a Computation with one layer of sub-computations

    def __init__(self, base_computation: Computation):
        self.base_computation = base_computation
        self.iterations: list = []

    @property
    def command(self) -> Command:
        return self.base_computation.command

    def __iter__(self) -> Generator[Computation]:
        if len(self.iterations) == 0:
            yield self.base_computation

        for iteration in enumerate(self.iterations):
            yield self.base_computation | iteration  # TODO

    def validate(self) -> bool:
        return True  # TODO

    def extend_computation(self, *args, **kwargs) -> list[Computation]:
        return [comp for comp in self]

    def parse_results(self, computation: Computation, results: list) -> dict:
        # Follows the interface of EMT
        """
        Parses the results obtained from an iterator obtained through extend_computation().
        :param computation: The computation asked by the upper layer
        :param results: The results for the list of computations obtained through extend_computation()
        :return:
        """
        res = {"results_list": []}
        n_shots = None
        for i, result in enumerate(results):
            result["iteration"] = self.iterations[i]
            res["results_list"].append(result)
            if "nshots" in result:  # TODO: see exact key - should be a constant somewhere
                n_shots = result["nshots"] if n_shots is None else n_shots + result["nshots"]

        return res
