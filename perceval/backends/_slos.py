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

from ._abstract_backends import AStrongSimulationBackend
from perceval.utils import Matrix, BasicState, BSDistribution, StateVector
from perceval.utils.logging import get_logger, channel

import exqalibur as xq
import math


class _Path:
    """A `Path` is the minimal computing graph for covering a set of input states"""

    def __init__(self, n, m, states, targets, backend):
        self._n = n
        self._m = m
        self._backend = backend
        self.coefs = Matrix.zeros((backend._mk_l[n], 1), use_symbolic=self._backend._symb)
        if n == 0:
            self.coefs.fill(1)

        if targets is None:
            targets = [list(state) for state in states]
        self._targets = []
        self._states = []
        self._children = {}
        for t, s in zip(targets, states):
            if sum(t) == 0:
                backend._state_mapping[s] = self
            else:
                self._targets.append(t)
                self._states.append(s)
        self._decompose()

    def _decompose(self):
        targets = self._targets
        states = self._states
        while targets:
            counts = [0] * len(targets[0])
            for one_target in targets:
                counts = [x + y for (x, y) in zip(counts, one_target)]
            max_mode = max(counts)
            max_index = counts.index(max_mode)
            current_targets = []
            current_states = []
            new_targets = []
            new_states = []
            for one_target, one_state in zip(targets, states):
                if one_target[max_index]:
                    one_target[max_index] -= 1
                    current_targets.append(one_target)
                    current_states.append(one_state)
                else:
                    new_targets.append(one_target)
                    new_states.append(one_state)
            self._children[max_index] = _Path(self._n + 1, self._m,
                                              current_states, current_targets,
                                              self._backend)
            targets = new_targets
            states = new_states

    def compute(self, u, parent_coefs: Matrix = None, mk: int = None):
        r"""Given the precompiled compute path, update all the coefficients"""
        if parent_coefs is not None:
            if self._backend._symb:
                self.coefs.fill(0)
                for parent_idx, coef_parent in enumerate(parent_coefs):
                    for j in range(self._m):
                        idx = self._backend._fsms[self._n].get(parent_idx, j)
                        if idx != xq.npos:
                            self.coefs[idx] += coef_parent * u[j, mk]
            else:
                self._backend._fsms[self._n].compute_slos_layer(u, self._m, mk, self.coefs, parent_coefs)

        for mk, child in self._children.items():
            child.compute(u, self.coefs, mk)


class SLOSBackend(AStrongSimulationBackend):
    def __init__(self, mask=None, n=None, use_symbolic=False):
        super().__init__()
        self._reset()
        self._symb = use_symbolic
        if mask is not None:
            self.set_mask(mask)
        if n is not None:
            get_logger().warn(
                f"DeprecationWarning: 'n' parameter is now ignored and deprecated. version=0.12", channel.user)

    @property
    def name(self) -> str:
        return "SLOS"

    def _reset(self):
        self._fsms: list[xq.FSMap] = [[]]
        self._fsas: dict[int, xq.FSArray] = {}
        self._mk_l: list[int] = [1]
        self._path_roots: list[_Path] = []
        self._state_mapping: dict[BasicState, _Path] = {}
        self._mask: xq.FSMask = None
        self.clear_iterator_cache()

    def _compute_path(self, umat):
        for path in self._path_roots:
            path.compute(umat)

    def set_circuit(self, circuit):
        previous_circuit = self._circuit
        assert not circuit.requires_polarization, "Circuit must not contain polarized components"
        self._input_state = None
        self._circuit = circuit
        self._umat = circuit.compute_unitary(use_symbolic=self._symb)
        if self._path_roots and previous_circuit.m == circuit.m:
            # Use the previously deployed paths to store the new circuit's coefs
            get_logger().debug("SLOS: compute coefficients keeping the previous path", channel.general)
            self._compute_path(self._umat)
        else:
            self._reset()

    def set_input_state(self, input_state: BasicState):
        super().set_input_state(input_state)
        self.preprocess([input_state])

    def clear_mask(self):
        super().clear_mask()
        self._reset()

    def _deploy(self, input_list: list[BasicState]):
        # allocate the fsas and fsms for covering all the input_states respecting possible mask
        # after calculation, we only need to keep fsa for input_state n
        # during calculation we need to keep current fsa and previous fsa
        m = self._circuit.m
        current_fsa = xq.FSArray(m, 0) if len(self._fsas) == 0 else self._fsas[max(self._fsas.keys())]
        for input_state in input_list:
            n = input_state.n
            if n < len(self._fsms) and n not in self._fsas:
                # we are missing the intermediate states - let us retrieve/load it back
                current_fsa = xq.FSArray(m, n, self._mask) if self._mask else xq.FSArray(m, n)
            for k in range(len(self._fsms), n + 1):
                fsa_n_m1 = current_fsa
                current_fsa = xq.FSArray(m, k, self._mask) if self._mask else xq.FSArray(m, k)
                self._mk_l.append(current_fsa.count())
                self._fsms.append(xq.FSMap(current_fsa, fsa_n_m1, True))
            if n not in self._fsas:
                self._fsas[n] = current_fsa

    def preprocess(self, input_list: list[BasicState]) -> bool:
        # now check if we have a path for the input states
        found_new = False
        for input_state in input_list:
            found_new = (input_state not in self._state_mapping)
            if found_new:
                break
        if not found_new:
            return False

        get_logger().debug("SLOS: deploy a new path and compute coefficients", channel.general)
        self._deploy(input_list)  # build the necessary fsa/fsms
        new_path = _Path(0, self._circuit.m, input_list, None, self)
        new_path.compute(self._umat)
        self._path_roots.append(new_path)
        return True

    def prob_amplitude(self, output_state: BasicState) -> complex:
        if self._input_state.n != output_state.n:
            return complex(0)
        output_idx = self._fsas[output_state.n].find(output_state)
        assert output_idx != xq.npos
        result = self._state_mapping[self._input_state].coefs[output_idx, 0] * math.sqrt(output_state.prodnfact() / self._input_state.prodnfact())
        return result if self._symb else complex(result)

    def prob_distribution(self) -> BSDistribution:
        istate = self._input_state
        c = self._state_mapping[istate].coefs.reshape(self._fsas[istate.n].count())
        c = abs(c) ** 2 / istate.prodnfact()
        xq.all_prob_normalize_output(c, self._fsas[istate.n])
        bsd = BSDistribution()
        for output_state, probability in zip(self._get_iterator(self._input_state), c):
            bsd.add(output_state, probability)
        return bsd

    def all_prob(self, input_state: BasicState = None):
        """SLOS specific signature, to enhance optimization in some computations"""
        if input_state is not None:
            self.set_input_state(input_state)
        else:
            input_state = self._input_state
        c = self._state_mapping[input_state].coefs.reshape(self._fsas[input_state.n].count())
        c = abs(c)**2 / self._input_state.prodnfact()
        xq.all_prob_normalize_output(c, self._fsas[input_state.n])
        return c

    def evolve(self) -> StateVector:
        istate = self._input_state
        c = self._state_mapping[istate].coefs.reshape(self._fsas[istate.n].count())
        res = StateVector()
        iprodnfact = istate.prodnfact()
        for output_state, pa in zip(self._get_iterator(self._input_state), c):
            res += output_state * (pa * math.sqrt(output_state.prodnfact() / iprodnfact))
        return res
