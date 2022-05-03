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

from abc import ABC, abstractmethod
import logging
import random
from typing import List, Tuple, Union, Iterator, Optional

from perceval.utils import Matrix, StateVector, AnnotatedBasicState, BasicState
from perceval.utils.statevector import convert_polarized_state, build_spatial_output_states
from ..components.circuit import ACircuit, _matrix_double_for_polarization

import quandelibc as qc
import numpy as np


class Backend(ABC):
    _name = None
    supports_symbolic = None
    supports_circuit_computing = None

    def __init__(self,
                 cu: Union[ACircuit, Matrix],
                 use_symbolic: bool = None,
                 use_polarization: Optional[bool] = None,
                 n: int = None,
                 mask: list = None):
        r"""
        :param cu: a circuit to simulate or a unitary Matrix symbolic or numeric
        :param use_symbolic: define if calculation should be symbolic or numeric:
            - None: decides based on U nature and backend capacity
            - True: calculation will be symbolic
            - False: calculation will be numeric
        :param n: expected number of input photons, necessary for applying masks
        :param mask: a mask for output states that we are interested in
        """
        self._logger = logging.getLogger(self._name)
        if not self.supports_circuit_computing:
            if isinstance(cu, ACircuit):
                if cu.requires_polarization:
                    if use_polarization is None:
                        use_polarization = True
                    else:
                        assert use_polarization, "use polarization can not be False for circuit with polarization"
                elif use_polarization is None:
                    use_polarization = False
                u = cu.compute_unitary(use_symbolic, use_polarization=use_polarization)
            else:
                if use_polarization is None:
                    use_polarization = False
                u = cu
                assert u.ndim == 2 and u.shape[0] == u.shape[1], "simulator works on square matrix"
            self._requires_polarization = use_polarization
            self._C = None
            if not self.supports_symbolic:
                if not u.defined or use_symbolic:
                    assert not u.is_symbolic, "%s backend does not support symbolic calculation" % self._name
                if not use_symbolic:
                    self._U = u.tonp()
            else:
                if use_symbolic and not u.is_symbolic:
                    self._U = Matrix(u, use_symbolic=True)
                elif use_symbolic is False and u.is_symbolic:
                    self._U = u.tonp()
                else:
                    use_symbolic = u.is_symbolic()
                    self._U = u
            self._realm: int = u.shape[0]
            self._m: int = self._requires_polarization and u.shape[0] >> 1 or u.shape[0]
            "number of modes"
        else:
            assert isinstance(cu, ACircuit),\
                "Component Based simulation works on circuit"
            assert not use_symbolic or self.supports_symbolic,\
                "%s backend does not support symbolic calculation" % self._name
            # component based simulation - we keep the circuit
            self._U = None
            self._C = cu
            self._m = self._realm = cu.m

        self._use_symbolic = use_symbolic
        self._n: int = n
        "number of photons - is required when using a mask"

        if mask is not None:
            assert n is not None, "number of photons required when using a mask"
            self._mask = qc.FSMask(self._m, n, mask)
        else:
            self._mask = None

        self._compiled_input = None

    def _changed_unitary(self, prev_u) -> None:
        """Notify change of unitary - might be used for backend with compiled states
        """
        return

    @property
    def is_symbolic(self):
        return self._U.is_symbolic

    @property
    def m(self):
        return self._m

    @property
    def U(self):
        return self._U

    @U.setter
    def U(self, u):
        prev_u = self._U
        self._U = u
        self._changed_unitary(prev_u)

    @abstractmethod
    def prob_be(self, input_state, output_state, n=None):
        raise NotImplementedError

    def probampli_be(self, input_state, output_state, n=None):
        raise NotImplementedError

    def prob(self,
             input_state: AnnotatedBasicState,
             output_state: AnnotatedBasicState,
             n: int = None,
             skip_compile: bool = False) -> float:
        r"""
        gives the probability of an output state given an input state
        :param input_state: the input state
        :param output_state: the output state
        :param n:
        :return: float probability
        """
        if input_state.n == 0:
            return output_state.n == 0
        if self._U is None or (not self._requires_polarization and not input_state.has_polarization):
            if hasattr(input_state, "separate_state"):
                input_states = hasattr(input_state, "separate_state") and input_state.separate_state() or [input_state]
                all_prob = 0
                for p_output_state in AnnotatedBasicState(output_state).partition(
                        [input_state.n for input_state in input_states]):
                    prob = 1
                    for i_state, o_state in zip(input_states, p_output_state):
                        if not skip_compile:
                            self.compile(i_state)
                        prob *= self.prob_be(i_state, o_state, n)
                    all_prob += prob
                return all_prob
            if not skip_compile:
                self.compile(input_state)
            return self.prob_be(input_state, output_state, n)
        spatial_mode_input_state, prep_matrix_input = convert_polarized_state(input_state)
        _U_ref = self._U
        _realm_ref = self._realm
        if not self._requires_polarization:
            _U_new = _matrix_double_for_polarization(self._m, self._U)
            self._realm = 2 * self._realm
        else:
            _U_new = self._U
        _U_new = _U_new @ prep_matrix_input
        if isinstance(output_state, AnnotatedBasicState) and output_state.has_polarization:
            # if output state is polarized, we will directly calculating probabilities for it
            spatial_mode_output_state, un_prep_matrix_output = convert_polarized_state(output_state, inverse=True)
            self.U = un_prep_matrix_output @ _U_new
            self.compile(spatial_mode_input_state)
            prob = self.prob_be(spatial_mode_input_state, spatial_mode_output_state, n)
        else:
            # for each polarized mode with k photons, we have to calculate probabilities on the spatial mode, ies all
            # |m,l> for m+l = k
            self.U = _U_new
            self.compile(spatial_mode_input_state)
            prob = 0
            for spatial_output in build_spatial_output_states(output_state):
                prob += self.prob_be(spatial_mode_input_state, spatial_output)
        self._U = _U_ref
        self._realm = _realm_ref
        return prob

    def all_prob(self, input_state: BasicState) -> np.ndarray:
        allprobs = []
        for(output, prob_output) in self.allstateprob_iterator(input_state):
            allprobs.append(prob_output)
        return np.asarray(allprobs)

    def probampli(self,
                  input_state: AnnotatedBasicState,
                  output_state: AnnotatedBasicState,
                  n: int = None) -> complex:
        """Gives the probability amplitude of an output state given an input state

        :param input_state: the input state
        :param output_state: the output state
        :param n:
        :return: complex probability amplitude
        """
        if input_state.n == 0:
            return output_state.n == 0
        if self._U is None or (not self._requires_polarization and not input_state.has_polarization):
            self.compile(input_state)
            return self.probampli_be(input_state, output_state, n)
        spatial_mode_input_state, prep_matrix_input = convert_polarized_state(input_state)
        _U_ref = self._U
        _realm_ref = self._realm
        if not self._requires_polarization:
            self._U = _matrix_double_for_polarization(self._m, self._U)
            self._realm = 2 * self._realm
        self.compile(spatial_mode_input_state)
        self._U = self._U @ prep_matrix_input
        if isinstance(output_state, AnnotatedBasicState) and output_state.has_polarization:
            # if output state is polarized, we will directly calculate probabilities for it
            spatial_mode_output_state, un_prep_matrix_output = convert_polarized_state(output_state, inverse=True)
            self._U = un_prep_matrix_output @ self._U
            prob_ampli = self.probampli_be(spatial_mode_input_state, spatial_mode_output_state, n)
        else:
            # for each polarized mode with k photons, we have to calculate probabilities on the spatial mode, ies all
            # |m,l> for m+l = k
            prob_ampli = 0
            for spatial_output in build_spatial_output_states(output_state):
                prob_ampli += self.probampli_be(spatial_mode_input_state, spatial_output)
        self._U = _U_ref
        self._realm = _realm_ref
        return prob_ampli

    def allstateprob_iterator(self,
                              input_state: Union[AnnotatedBasicState, StateVector]) \
            -> Iterator[Tuple[AnnotatedBasicState, float]]:
        """Iterator on all possible output states compatibles with mask generating (`StateVector`, probability)

        :param input_state: a given input state
        :return: list of (output_state, probability)
        """
        skip_compile = False
        for output_state in self.allstate_iterator(input_state):
            if isinstance(input_state, StateVector) and len(input_state) > 1:
                # a superposed state cannot have distinguishable particles
                probampli = 0
                sv = input_state
                for inp_state in sv:
                    probampli += self.probampli(inp_state, output_state)*sv[inp_state]
                yield output_state, abs(probampli)**2
            else:
                # TODO: should not have a special case here
                if isinstance(input_state, StateVector):
                    input_state = input_state[0]
                yield output_state, self.prob(input_state, output_state, skip_compile=skip_compile)
                skip_compile = True

    def allstate_iterator(self, input_state: Union[AnnotatedBasicState, StateVector]) -> AnnotatedBasicState:
        """Iterator on all possible output states compatible with mask generating StateVector

        :param input_state: a given input state vector
        :return: list of output_state
        """
        m = self.m
        ns = input_state.n
        if not isinstance(ns, list):
            ns = [ns]
        for n in ns:
            if self._mask:
                output_array = qc.FSArray(m, n, self._mask)
            else:
                output_array = qc.FSArray(m, n)
            for output_idx, output_state in enumerate(output_array):
                yield AnnotatedBasicState(output_state)

    def evolve(self, input_state: [AnnotatedBasicState, StateVector]) -> StateVector:
        r"""StateVector evolution through a circuit

        :param input_state: the input_state
        :return: the output_state
        """
        output_state = StateVector(None)
        for basic_output_state in self.allstate_iterator(input_state):
            if isinstance(input_state, StateVector):
                sv = input_state
                for inp_state in sv:
                    output_state[basic_output_state] += self.probampli(inp_state, basic_output_state)*sv[inp_state]
            else:
                output_state[basic_output_state] += self.probampli(input_state, basic_output_state)
        return output_state

    def compile(self,
                input_states: Union[StateVector, List[StateVector]]) -> bool:
        """
        Compile a simulator to work with one or specific input_states - might do nothing for some backends
        :param input_states: list of input states
        :return: True if any compilation happened, False otherwise
        """
        return False

    def sample(self, input_state):
        prob = random.random()
        output_state = None
        for (output_state, state_prob) in self.allstateprob_iterator(input_state):
            if state_prob >= prob:
                return output_state
            prob -= state_prob
        return output_state
