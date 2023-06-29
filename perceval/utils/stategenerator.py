from . import StateVector
from . import BasicState
from enum import Enum
import networkx as nx

class Encoding(Enum):
    DUAL_RAIL = 0
    POLARIZATION = 1
    QUDIT = 2
    TIME = 3
    RAW = 4

class StateGenerator:
    r"""
    StateGenerator class for conveniently generating common complex StateVectors

    :param encoding: for specifying the output format of the StateVector
        supported are Encoding.RAW, Encoding.DUAL_RAIL, Encoding.POLARIZATION
    :param polarization_base: (optional) you can provide your own polarization basis as a tuple of BasicStates
        default=(BasicState("|{P:H}>"), BasicState("|{P:V}>")
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
        r"""
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
        r"""
        Generate a StateVector representing a Bell state

        :param state: name of the bell state you want to generate:
            \"phi+\" = (|0,0>+|1,1>)/sqrt(2)
            \"phi-\" = (|0,0>-|1,1>)/sqrt(2)
            \"psi+\" = (|0,1>+|1,0>)/sqrt(2)
            \"psi-\" = (|0,1>-|1,0>)/sqrt(2)
        :return: StateVector for a bell state
        """

        if state == "phi+":
            sv = StateVector(self._zero_state ** 2) + StateVector(BasicState(self._one_state ** 2))
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
        r"""
        Generate a StateVector representing a (generalized) Greenberger–Horne–Zeilinger state

        :param n: order of the GHZ state
        :return: StateVector representing the GHZ state
        """
        assert n>2, "A (generalized) Greenberger–Horne–Zeilinger state is only defined for n>2"
        sv = StateVector(self._zero_state ** n) + StateVector(self._one_state ** n)
        return sv

    def graph_state(self, graph: nx.Graph):
        r"""
        Generate a StateVector representing a graph state.

        :param graph: networkx.Graph object. Edge weights are ignored.
        :return: StateVector representing the graph state
        """
        sv = StateVector()

        if graph.number_of_nodes() == 0:
            return sv

        basicstates = [self._one_state, self._zero_state]

        # generate all basic states
        for i in range(1,graph.number_of_nodes()):
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

        return sv