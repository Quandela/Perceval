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

import networkx as nx
import numpy as np

from .statevector import BasicState, StateVector
from .qmath import distinct_permutations
from ._enums import Encoding
from .logging import get_logger, channel


class StateGenerator:
    """
    StateGenerator class for conveniently generating common complex StateVectors

    :param encoding: for specifying the output format of the StateVector
        supported are Encoding.RAW, Encoding.DUAL_RAIL, Encoding.POLARIZATION
    :param polarization_base: (optional) you can provide your own polarization basis as a tuple of BasicStates.
        default=(BasicState("`|{P:H}>`"), BasicState("`|{P:V}>`")
    """
    def __init__(self, encoding, polarization_base=(BasicState("|{P:H}>"), BasicState("|{P:V}>"))):

        assert isinstance(encoding, Encoding), "You need to provide an encoding"
        if encoding == Encoding.RAW:
            self._zero_state = BasicState("|0>")
            self._one_state = BasicState("|1>")
        elif encoding == Encoding.DUAL_RAIL:
            self._zero_state = BasicState("|1,0>")
            self._one_state = BasicState("|0,1>")
        elif encoding == Encoding.POLARIZATION:
            if len(polarization_base[0]) != 1 or len(polarization_base[1]) != 1:
                raise ValueError("The BasicStates representing the polarization basis should only contain one mode")
            self._zero_state = polarization_base[0]
            self._one_state = polarization_base[1]
        else:
            raise ValueError("Only use RAW, DUAL_RAIL or POLARIZATION encoding.")

    def logical_state(self, state: list[int]):
        """
        Generate a StateVector from a list of logical state

        :param state: list of bits
        :return: StateVector representing the logical state
        """

        sv = StateVector()
        for bit in state:
            if bit == 0:
                sv = sv * self._zero_state
            elif bit == 1:
                sv = sv * self._one_state
            else:
                raise ValueError("The argument list corresponding to a logical state should only contain 0s and 1s")

        return sv

    def bell_state(self, state: str):
        """
        Generate a StateVector representing a Bell state

        :param state: name of the bell state you want to generate:

            * "phi+" = :math:`\\frac{|0,0>+|1,1>}{\sqrt{2}}`
            * "phi-" = :math:`\\frac{|0,0>-|1,1>}{\sqrt{2}}`
            * "psi+" = :math:`\\frac{|0,1>+|1,0>}{\sqrt{2}}`
            * "psi-" = :math:`\\frac{|0,1>-|1,0>}{\sqrt{2}}`
        :return: StateVector for a bell state
        """

        if state == "phi+":
            sv = StateVector(self._zero_state ** 2) + StateVector(self._one_state ** 2)
            return sv
        elif state == "phi-":
            sv = StateVector(self._zero_state ** 2) - StateVector(self._one_state ** 2)
            return sv
        elif state == "psi+":
            sv = StateVector(self._zero_state * self._one_state) + StateVector(self._one_state * self._zero_state)
            return sv
        elif state == "psi-":
            sv = StateVector(self._zero_state * self._one_state) - StateVector(self._one_state * self._zero_state)
            return sv

        raise ValueError("The state parameter must contain one of the Bell states as a string: phi+,phi-,psi+,psi-")

    def ghz_state(self, n: int):
        """
        Generate a StateVector representing a (generalized) Greenberger-Horne-Zeilinger state

        :param n: order of the GHZ state
        :return: StateVector representing the GHZ state
        """
        assert n > 2, "A (generalized) Greenberger-Horne-Zeilinger state is only defined for n>2"
        sv = StateVector(self._zero_state ** n) + StateVector(self._one_state ** n)
        return sv

    def graph_state(self, graph: nx.Graph):
        """
        Generate a StateVector representing a graph state.

        :param graph: networkx.Graph object. Edge weights are ignored.
        :return: StateVector representing the graph state
        """
        sv = StateVector()

        if graph.number_of_nodes() == 0:
            return sv

        basicstates = [self._one_state, self._zero_state]

        # generate all basic states
        for _ in range(1, graph.number_of_nodes()):
            for j in range(len(basicstates)):
                basicstates.append(basicstates[j] * self._one_state)
                basicstates[j] = basicstates[j] * self._zero_state

        # calculate signum of each BasicState and add it to the result StateVector (corresponding to Controlled Z Gate)
        for bs in basicstates:
            sgn = 1
            for u, v in graph.edges:
                enclen = len(self._zero_state)
                posu = u * enclen
                posz = v * enclen

                if bs[posu:posu + enclen] == self._one_state and bs[posz:posz + enclen] == self._one_state:
                    sgn = -1*sgn

            if sgn == -1:
                sv = sv - StateVector(bs)
            else:
                sv = sv + StateVector(bs)
        sv.normalize()
        return sv

    def dicke_state(self, n: int, k: int = None) -> StateVector:
        """Get the Dicke state :math:`|D(n,k)>` which is the equal superposition state of all :math:`C(n,k)` basis states of weight :math:`k`

        Mode number:
            * For RAW and Polarization: :math:`n`
            * For Dual rail encoding: :math:`2*n`

        Photon number:
            * For Raw encoding: :math:`k`
            * For Dual rail and Polarization encoding: :math:`n`

        :param n: Number of qubits equal to :math:`|1>_L` or photons
        :param k: Weight (Number of qubits or modes)
        :return: Dicke state vector
        """

        if not isinstance(n, int):
            raise TypeError(f"n parameter should be an int and not {type(n)}")

        if n < 1:
            raise ValueError("Only support strictly positive one state number")

        if not k:
            k = 2*n
        else:
            if not isinstance(k, int):
                raise TypeError(f"k parameter should be an int and not {type(k)}")
            if k < n:
                get_logger().warn(f"Generating an empty state since {k} is smaller than {n}", channel.user)
                return StateVector()

        dicke_state = StateVector()
        array = [str(self._one_state)[1:-1]]*n + [str(self._zero_state)[1:-1]]*(k-n)
        for state in distinct_permutations(array):
            dicke_state += BasicState(f"|{','.join(state)}>")
        return dicke_state

    @staticmethod
    def zero_padded_state(n: int, m: int = None) -> BasicState:
        """
        Generate a |111...10...0> BasicState with n photons and m modes. The result is independent of the encoding.

        :param n: Number of photons
        :param m: Number of modes. Default :math:`n`
        :return: BasicState of the shape |111...10...0>
        """
        if m is None:
            return BasicState(n * [1])
        assert n <= m, "Cannot generate a zero-padded state with more photons than modes"
        return BasicState(n * [1] + (m - n) * [0])

    @staticmethod
    def periodic_state(n: int, m: int = None) -> BasicState:
        """
        Generate a BasicState consisting of repeating |10> n times. Pads the end of the state with zero photon modes.
        The result is independent of the encoding.

        :param n: Number of photons
        :param m: Number of modes. Default :math:`2n` (no padding at the end)
        :return: BasicState of the shape |1010...000>
        """
        state = BasicState([1, 0] * n)
        if m is not None:
            assert 2 * n <= m, "Cannot generate a periodic state with a number of modes less than twice the number of photons"
            state *= BasicState([0]) ** (m - state.m)
        return state

    @staticmethod
    def evenly_spaced_state(n: int, m: int) -> BasicState:
        """
        Generate a BasicState where photons are evenly spaced. The result is independent of the encoding.

        :param n: Number of photons
        :param m: Number of modes.
        :return: BasicState where the photons are as spread out as possible in the modes.
        """
        res = [0] * m
        if n == 1:
            # Special case: put the photon in the middle
            res[m // 2] = 1
        else:
            # 1e-5 induces a very small bias but avoids getting m in the positions, which would be out of range
            positions = np.linspace(0, m - 1e-5, n, dtype=int)
            for pos in positions:
                res[pos] += 1

        return BasicState(res)
