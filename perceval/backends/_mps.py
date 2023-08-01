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

from typing import Union
import copy

import numpy as np
from math import factorial
from scipy.special import comb
from collections import defaultdict

from ._abstract_backends import AProbAmpliBackend
from perceval.utils import BasicState
from perceval.components import ACircuit


class MPSBackend(AProbAmpliBackend):
    """Step-by-step circuit propagation algorithm, works on a circuit.
    Approximate the probability amplitudes with a cutoff.
    - For now only supports Phase shifters and Beam Splitters
    - TODO: link to the quandelibc computation

    -Raksha
    Adding comments to understand and fix MPS. The basic algorithm
    - The input state vector is approximated by a Matrix Product State (MPS)
    - Step-by-step circuit propagation refers to the application of individual
    components of the circuit on MPS - updating it until the output MPS is found
    - The different prob ampli coefficients of the possible output states are then
    computed from the final MPS
    """

    def __init__(self):
        super().__init__()
        self._s_min = 1e-8
        self._cutoff = None
        self._compiled_input = None
        self._current_input = None
        self._res = defaultdict(lambda: defaultdict(lambda: np.array([0])))
        # _res stores output of the state compilation in MPS.
        # It is a Nested DefaultDict. The outermost dict has "input_states"
        # as keys with values=DefaultDict. We can do multiple computation
        # for different input_states.
        # The 2nd (nested inside) DefaultDict has 2 keys "gamma" and "sv". They
        # represent the matrices $\Gamma$ and $\lambda$ of the MPS for a
        # given input_state. Their values are numpy arrays containing the full MPS.

    @property
    def name(self) -> str:
        return "MPS"

    def set_cutoff(self, cutoff_val: int):
        """
        This parameter defines the Schmidt rank of the decomposition of the
        state into an MPS, in other words - how well approximated the state is.
        """
        assert isinstance(cutoff_val, int), "cutoff must be an integer"
        self._cutoff = cutoff_val

    def set_circuit(self, circuit: ACircuit):
        super().set_circuit(circuit)
        C = self._circuit
        for r, c in C:
            assert c.compute_unitary(use_symbolic=False).shape[0] <= 2, \
                "MPS backend can not be used with components of using more than 2 modes"
        if self._cutoff is None:
            self._cutoff = C.m  # sets the default value of cut-off = Num of modes of circuit
        # self._clear_cache()  # todo: _clear_cache() should define _res? Discuss: Eric

    def set_input_state(self, input_state: BasicState):
        super().set_input_state(input_state)
        self.compile()

    def apply(self, r, c):
        """
        Applies the components of the circuit iteratively to update the MPS.

        :param r: List of the mode positions for a component of the Circuit
        :param c: The component
        """
        u = c.compute_unitary(False)
        k_mode = r[0]  # k-th mode is where the upper mode(only) of the BS(PS) component is connected
        if len(u) == 2:  # BS
            self.update_state_2_mode(k_mode, u)  # --> quandelibc
        elif len(u) == 1:  # PS
            self.update_state_1_mode(k_mode, u)  # --> quandelibc

    def compile(self) -> bool:
        C = self._circuit
        var = [float(p) for p in C.get_parameters()]  # to check if the state is already computed
        if self._compiled_input and self._compiled_input[0] == var and self._input_state in self._res:
            return False
        self._compiled_input = copy.copy((var, self._input_state))
        self._current_input = None  # todo: verify - do I need to set it to None again?

        # TODO : allow any StateVector as in stepper, or a list as in SLOS?
        # self._input_state *= BasicState([0] * (self._input_state.m - self._input_state.m))
        self._n = self._input_state.n  # number of photons
        self._d = self._n + 1  # possible num of photons in each mode {0,1,2,...,n}
        self._cutoff = min(self._cutoff, self._d ** (self._input_state.m//2))
        # choosing a cut-off smaller than the limit as the size of matrix increases
        # exponentially with cutoff
        # this is the Schmidt's rank or bond dimension ($\chi$ in Thibaud's notes)

        self._gamma = np.zeros((self._input_state.m, self._cutoff, self._cutoff, self._d), dtype='complex_')
        # Gamma matrices of the MPS - array shape (m, $\chi$, $\chi$, d)
        for i in range(self._input_state.m):
            self._gamma[i, 0, 0, self._input_state[i]] = 1

        self._sv = np.zeros((self._input_state.m, self._cutoff))
        # sv matrices are diagonal matrices with singular values - array shape (m,$\chi$)
        self._sv[:, 0] = 1  # first column set to 1
        # Todo: understand completely the initialization of gamma and sv above

        for r, c in C:
            # r -> tuple -> lists the modes where the component c is connected
            self.apply(r, c)

        self._res[tuple(self._input_state)]["gamma"] = self._gamma.copy()
        self._res[tuple(self._input_state)]["sv"] = self._sv.copy()

        return True

    def prob_amplitude(self, output_state: BasicState) -> complex:
        """
        This takes in the expected output states, reads the input and from
        self._res extracts the correponding gamma and diagonal sv matrices.
        All of this goes to mps_in_list -> each element in order is gamma-sv-gamma-sv-...
        Returns the full contraction -> multidot of all -> which I expect to be the tensor
        containing the prob amplitude coefficients |psi> = c_tensor |pure statevectors>
        """
        # TODO: put in quandelibc
        m = self._input_state.m
        mps_in_list = []
        self._current_input = tuple(self._input_state)
        for k in range(m - 1):
            mps_in_list.append(self._res[tuple(self._input_state)]["gamma"][k, :, :, output_state[k]])
            # _res[1ST LEVEL: selects dict -> given input state][2ND LEVEL: selects "gamma" matrix key of that]
            # [3RD LEVEL: gamma is np.array -> this indexing chooses corresponding output needed]
            # seems to be but confused
            mps_in_list.append(self._sv_diag(k))
            # alternately takes in each gamma and singular value matrices -> puts them in a list
        mps_in_list.append(self._res[tuple(self._input_state)]["gamma"][self._input_state.m-1, :, :, output_state[self._input_state.m-1]])
        # todo : understand how the multidot would act on this list to return the prob amplis
        # my guess is that it does the full tensor contraction -> outer product of all
        # different terms of MPS
        return np.linalg.multi_dot(mps_in_list)[0, 0]

    @staticmethod
    def preferred_command() -> str:
        return 'probampli'

# ################ From here, everything must be in quandelibc ##############################

    def _transition_matrix_1_mode(self, u):
        """
        transition matrix "U" related to the application of a phase shifter one a single mode.

        size of this "U" depends on the possible number of photons {0,1,2,...n} ==> d=n+1.
        returns the full transition matrix "U" related to the component that will update the
        corresponding mode's "gamma" of the matrix product state
        """
        d = self._d
        big_u = np.zeros((d, d), dtype='complex_')
        for i in range(d):
            big_u[i, i] = u[0, 0] ** i
        return big_u

    def update_state_1_mode(self, k, u):
        """
        tensor contraction between the corresponding mode's "$\Gamma$" of the MPS
        and the transition matrix "U" of phase shifter for that mode [_transition_matrix_1_mode].
        """
        self._gamma[k] = np.tensordot(self._gamma[k], self._transition_matrix_1_mode(u), axes=(2, 0))
        # gamma[k] -> takes the kth slice from first dimension -> selects gamma of kth mode
        # the shape of gamma[k] is ($\chi$, $\chi$, d).
        # The shape of the matrix returned by _transition_matrix_1_mode(u) is (d, d).
        # The contraction is on the free index 'd' here, hence 2 of gamma[k]
        # and 0 of the transition matrix. Assigns the result to the same gamma[k]
        # and in the process update it.

    def update_state_2_mode(self, k, u):
        """
        takes the gamma->kth and (k+1)th + corresponding $\lambda$-s -> contracts the entire thing
        with 2 mode beam splitter, performs some reshaping and then svd to re-build the corresponding
        segment of MPS.
        todo: verify the axes in tensordots, imrpovement of code
        """

        if 0 < k < self._input_state.m - 2:
            # BS anywhere besides the first and the last mode
            theta = np.tensordot(self._sv_diag(k - 1), self._gamma[k, :], axes=(1, 0))
            # _sv_diag(k - 1) -> shape ($\chi$, $\chi$) and _gamma[k, :] of shape ($\chi$, $\chi$, d)
            # Output = theta of shape ($\chi$, $\chi$, d)
            theta = np.tensordot(theta, self._sv_diag(k), axes=(1, 0))
            # theta of shape ($\chi$, $\chi$, d) and _sv_diag(k - 1) -> shape ($\chi$, $\chi$)
            # Output = theta of shape ($\chi$, $\chi$, d)
            theta = np.tensordot(theta, self._gamma[k + 1, :], axes=(2, 0))
            # theta of shape ($\chi$, $\chi$, d) and _gamma[k+1, :] of shape ($\chi$, $\chi$, d)
            # doubt here with 2 todo: fix
            theta = np.tensordot(theta, self._sv_diag(k + 1), axes=(2, 0))
            # contraction of the corresponding matrices of MPS finished until here
            theta = np.tensordot(theta, self._transition_matrix_2_mode(u), axes=([1, 2], [0, 1]))
            # theta of shape ($\chi$, $\chi$, d) and _transition_matrix_2_mode(u) shape ()
            # should be full contraction of 2 edges of theta (previously connected to MPS, but why 2?)
            # todo: Rawad - I think 2 was for d index
            theta = theta.swapaxes(1, 2).swapaxes(0, 1).swapaxes(2, 3)
            theta = theta.reshape(self._d * self._cutoff, self._d * self._cutoff)
            # merging indices to have a 2D matrix of shape (d x $\chi$, d x $\chi$) -> step before SVD

        # the following 2 edge cases require one less tensordot/contraction as there would not be a
        # sv_diagonal available
        elif k == 0:
            # BS connected between the first 2 modes -> Edge of circuit
            theta = np.tensordot(self._gamma[k, :], self._sv_diag(k), axes=(1, 0))
            theta = np.tensordot(theta, self._gamma[k + 1, :], axes=(2, 0))
            theta = np.tensordot(theta, self._sv_diag(k + 1), axes=(2, 0))
            theta = np.tensordot(theta, self._transition_matrix_2_mode(u),
                                 axes=([1, 2], [0, 1]))  # Pretty weird thing... To check
            # todo: tensorproduct of mps with BS is going to
            #  include states that he probably neglected while building the BS. Verify
            theta = theta.swapaxes(1, 2).swapaxes(0, 1).swapaxes(2, 3)
            theta = theta.reshape(self._d * self._cutoff, self._d * self._cutoff)

        elif k == self._input_state.m - 2:
            # BS connected between the last 2 modes -> Edge of circuit
            theta = np.tensordot(self._sv_diag(k - 1), self._gamma[k, :], axes=(1, 0))
            theta = np.tensordot(theta, self._sv_diag(k), axes=(1, 0))
            theta = np.tensordot(theta, self._gamma[k + 1, :], axes=(2, 0))
            theta = np.tensordot(theta, self._transition_matrix_2_mode(u), axes=([1, 3], [0, 1]))
            theta = theta.swapaxes(1, 2).swapaxes(0, 1).swapaxes(2, 3)
            theta = theta.reshape(self._d * self._cutoff, self._d * self._cutoff)

        v, s, w = np.linalg.svd(theta)  # svd after all contractions to splits up the big theta matrix
        # in standard notation SVD is written as M=USV, but we keep 'u' for unitary,
        # this implies U->v, S->s, V->w

        v = v.reshape(self._d, self._cutoff, self._d * self._cutoff).swapaxes(0, 1).swapaxes(1, 2)[:, :self._cutoff]
        w = w.reshape(self._d * self._cutoff, self._d, self._cutoff).swapaxes(1, 2)[:self._cutoff]
        s = s[:self._cutoff]  # resticting the size of SV matrices to cut_off -> truncation

        self._sv[k] = np.where(s > self._s_min, s, 0)  # updating corresponding sv after the action of BS
        # _s_min is too low, do we really need this todo: ask Rawad

        # todo: below - seems weird, discuss Rawad
        if k > 0:
            rank = np.nonzero(self._sv[k - 1])[0][-1] + 1
            self._gamma[k, :rank] = v[:rank] / self._sv[k - 1, :rank][:, np.newaxis, np.newaxis]
            self._gamma[k, rank:] = 0
        else:
            self._gamma[k] = v
        if k < self._input_state.m - 2:
            rank = np.nonzero(self._sv[k + 1])[0][-1] + 1
            self._gamma[k + 1, :, :rank] = (w[:, :rank] / self._sv[k + 1, :rank][:, np.newaxis])
            self._gamma[k + 1, :, rank:] = 0
        else:
            self._gamma[k + 1] = w

    def _transition_matrix_2_mode(self, u):
        """
        this function computes the elements
        (I,J) = (i_k, i_k+1, j_k, j_k+1) of the matrix U_k,k+1.
        This is concerned with the action of beam splitter between given 2 modes.
        input parameter u is the unitary matrix of the Beam splitter - 2x2 matrix
        The formula for constructing the larger U to contract with the MPS is in
        Thibaud report.
        """
        d = self._d
        big_u = np.zeros((d, d, d, d), dtype='complex_')  # matrix corresponding to BS -> to contract with MPS
        # todo: vectorize and remove so many for loops
        # comment: another possible error - size of the big_u constructed, maybe it does not take all photons
        for i1 in range(d):  # i1=n1 in the formula in report
            for i2 in range(d):  # i2=n2 in the formula in report
                itot = i1 + i2
                u1, u2, u3, u4 = u[0,0], u[0,1], u[1, 0], u[1, 1]
                outputs = np.zeros((d,d), dtype = 'complex_')
                if itot <= self._n:  # cannot exceed the total number of photons
                    # todo: try removing this if and check : Stephen. possibly he is applying 0 to some state that exist
                    for k1 in range(i1+1):
                        for k2 in range(i2+1):
                            outputs[k1 + k2, itot - (k1 + k2)] += comb(i1, k1)*comb(i2, k2)\
                            *(u1**k1*u2**k2*u3**(i1-k1)*u4**(i2-k2))\
                            *(np.sqrt(factorial(k1+k2)*factorial(itot-k1-k2)))  # todo: verfiy; i think this is incorrect

                big_u[i1,i2,:] = outputs/(np.sqrt(factorial(i1)*factorial(i2)))
        return big_u

    def _sv_diag(self, k):
        """
        Creates the diagonal matrix containing the singular values of
        the matrices in the MPS
        todo: math behind to verify
        doubt with how the data is extracted and the sv_matrix is constructed
        particularly with choosing when to read from the self._res or
        from the sv initiated with single element = 1
        """
        if self._res[self._current_input]["sv"].any():
            sv = self._res[self._current_input]["sv"]
        else:
            sv = self._sv
        sv_diag = np.zeros((self._cutoff, self._cutoff))
        np.fill_diagonal(sv_diag, sv[k, :])
        return sv_diag
