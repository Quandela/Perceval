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

from perceval.components import ACircuit
from perceval.utils import BasicState, BSDistribution, StateVector, SVDistribution
from perceval.backends._abstract_backends import AProbAmpliBackend

from multipledispatch import dispatch



class Simulator:

    def __init__(self, backend: AProbAmpliBackend):
        self._backend = backend
        self.DEBUG_computation_count = 0

    def set_circuit(self, circuit: ACircuit):
        self._backend.set_circuit(circuit)

    @dispatch(BasicState)
    def probs(self, input_state: BasicState) -> BSDistribution:
        input_list = input_state.separate_state()
        input_set = set(input_list)
        self.DEBUG_computation_count = len(input_set)
        pdists = []
        for input_state in input_set:
            self._backend.set_input_state(input_state)
            pdists.append(self._backend.prob_distribution())
        results = BSDistribution()
        for input_state in input_list:
            idx = list(input_set).index(input_state)
            results = BSDistribution.tensor_product(results, pdists[idx])
        return results

    @dispatch(StateVector)
    def probs(self, input_state: StateVector):
        raise NotImplementedError()

    @dispatch(SVDistribution)
    def probs(self, input_state: SVDistribution):
        raise NotImplementedError()
