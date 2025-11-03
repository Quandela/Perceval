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

from packaging.version import Version
from perceval.components.linear_circuit import ACircuit
from perceval.utils.matrix import Matrix

class CompiledCircuit(ACircuit):
    def __init__(self, name: str, template_or_size: ACircuit | int, version: Version, parameters: list[float]):
        m = template_or_size if isinstance(template_or_size, int) else template_or_size.m
        template = template_or_size if isinstance(template_or_size, ACircuit) else None
        super().__init__(m, name)
        self.version = version
        self.parameters = parameters
        self._template = template
        if self._template:
            assert len(self._template.params) == len(self.parameters), "Incorrect BasicState size"

    def _compute_unitary(self,
                         assign: dict = None,
                         use_symbolic: bool = False) -> Matrix:
        """Compute the unitary matrix corresponding to the current circuit

        :param assign: assign values to some parameters
        :param use_symbolic: if the matrix should use symbolic calculation
        :return: the unitary matrix, will be a :class:`~perceval.utils.matrix.MatrixS` if symbolic, or a ~`MatrixN`
                 if not.
        """
        if not self._template:
            raise RuntimeError("Missing template to compute unitary for CompiledCircuit")
        for f, p in zip(self.parameters, self._template.get_parameters()):
            p.set_value(f)
        return self._template.compute_unitary(dict, use_symbolic)

    def describe(self) -> str:
        """
        Describe the component as the Python code that generates it.

        :return: code generating the component
        """
        pass
