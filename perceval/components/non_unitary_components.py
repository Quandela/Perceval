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
import numpy as np
import scipy as sc

from perceval.utils import BasicState, StateVector
from perceval.components.abstract_component import AParametrizedComponent


class TD(AParametrizedComponent):
    """Time delay"""
    DEFAULT_NAME = "TD"

    def __init__(self, dt):
        super().__init__(1)
        self._dt = self._set_parameter("t", dt, 0, None, False)

    def get_variables(self, map_param_kid=None):  # is this useful?
        parameters = {}
        if map_param_kid is None:
            map_param_kid = self.map_parameters()
        self.variable_def(parameters, "t", "t", None, map_param_kid)
        return parameters

    def describe(self):
        if self._dt.fixed:
            value = float(self._dt)
        else:
            value = f'P("{self._dt.spv}")'
        return f"TD(t={value})"


class LC(AParametrizedComponent):
    """Loss channel"""
    DEFAULT_NAME = "LC"

    def __init__(self, loss):
        super().__init__(1)
        self._loss = self._set_parameter("loss", loss, 0, 1, False)

    def get_variables(self, map_param_kid=None):
        parameters = {}
        if map_param_kid is None:
            map_param_kid = self.map_parameters()
        self.variable_def(parameters, "loss", "loss", None, map_param_kid)
        return parameters

    def describe(self):
        if self._loss.fixed:
            value = float(self._loss)
        else:
            value = f'P("{self._loss.spv}")'
        return f"LC(loss={value})"

    def apply(self, r, sv):
        """
        Applies a channel loss to r-th mode on an input StateVector sv
        Channel loss is treated as a beam splitter with a reflectivity equal to the loss. This beam splitter
        being connected to a "virtual" mode containing lost photons

        The output state vector contains BasicStates which are 1 mode bigger than the ones in input:
        (input modes count + the virtual mode)
        """
        # Assumes r of size 1
        # Returns a stateVector of size m + 1. Stepper backend should support this
        if isinstance(sv, BasicState):
            sv = StateVector(sv)

        r = r[0]
        loss = self.get_variables()["loss"]

        n_max = max(state[r] for state in sv)
        N = np.arange(n_max + 1)
        k = np.arange(n_max + 1)
        k = np.tile(k, (n_max+1, 1)).transpose()

        prob = sc.special.comb(np.tile(N, (n_max+1, 1)), k)
        prob *= loss ** (sc.sparse.diags([(n_max + 1 - i) * [i] for i in range(n_max + 1)],
                                         list(range(n_max + 1))).toarray())
        prob *= (1 - loss) ** k
        prob = np.sqrt(prob)

        nsv = StateVector()
        nsv.m = sv.m + 1
        # Equivalent to the nsv.update below:
        # for state, prob_ampli in sv.items():
        #     n = state[r]
        #     for i in range(n + 1):
        #         nsv[BasicState(state.set_slice(slice(r, r+1), BasicState([i]))) * BasicState([n - i])] += prob_ampli \
        #                                                                             * (loss ** (n - i)
        #                                                                                * (1 - loss) ** i
        #                                                                                * comb(n, i)) ** 0.5

        # Dict comprehension is possible here as two different basic states can't give the same resulting state
        nsv.update(
            {
                BasicState(state.set_slice(slice(r, r + 1), BasicState([i]))) * BasicState([state[r] - i]):
                    prob_ampli * prob[i, state[r]]
                for state, prob_ampli in sv.items()
                for i in range(state[r] + 1)
            }
        )
        return nsv
