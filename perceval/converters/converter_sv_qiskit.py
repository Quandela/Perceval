import numpy as np
from perceval.utils import StateVector
from perceval.utils import BasicState
from enum import Enum
from qiskit.quantum_info import Statevector as Qiskit_sv


class Encoding(Enum):
    DUAL_RAIL = 0
    POLARIZATION = 1
    QUDIT = 2
    TIME = 3
    RAW = 4


class StatevectorConverter:

    def __init__(self, encoding, polarization_base=(BasicState("|{P:H}>"), BasicState("|{P:V}>")), ancillae=None):
        r"""
        :param encoding: for specifying the output format of the StateVector
            supported are Encoding.RAW, Encoding.DUAL_RAIL, Encoding.POLARIZATION
        :param polarization_base:(optional) you can provide your own polarization basis as a tuple of BasicStates
            default=(BasicState("|{P:H}>"), BasicState("|{P:V}>")
        :param ancillae: (optional) you can  provide a list of additional modes, not taken in account for n-qubit
        """

        if ancillae is None:
            ancillae = []
        self.ancillae = ancillae

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

    def remove_ancilla(self, sv):
        r"""Removes the auxiliary modes to obtain a proper n-qubits state
        """

        ancillae = np.sort(self.ancillae)
        l_a = len(ancillae)
        new_sv = StateVector()
        for state in sv:
            bs = BasicState(state)
            new_bs = StateVector()
            previous = -1
            for i in range(l_a):
                # recreate each BasicState without the ancilla modes
                new_bs = new_bs * bs[previous + 1:ancillae[i]]
                previous = ancillae[i]
            new_sv = new_sv + sv[bs] * (new_bs * bs[ancillae[l_a - 1] + 1:])

        if len(sv) != len(new_sv):
            raise ValueError(
                "The StateVector doesn't represent a n-qubit: some terms have been suppressed while removing ancillae")
        else:
            sv = new_sv
        return sv

    def sv_to_qiskit(self, sv):
        r"""Converts a StateVector from perceval to a Statevector from qiskit
        """

        l_sv = len(sv)
        if l_sv == 0:
            raise ValueError("The StateVector is empty")
        if len(self.ancillae) != 0:
            sv = self.remove_ancilla(sv)

        zero, one = self._zero_state, self._one_state
        step = len(zero)

        l_bs = len(sv[0])

        if l_bs % step != 0:
            raise ValueError("The StateVector doesn't represent a n-qubit")
        else:
            l_n_qbt = l_bs // step

        ampli = np.zeros(2 ** l_n_qbt, dtype=complex)
        for state in sv:
            bs = BasicState(state)
            n = 0
            for i in range(l_n_qbt):
                # check the value of each qubit
                # i-th qubit = 1
                if bs[step * i: step * i + step] == one:
                    n += 2 ** (l_n_qbt - i - 1)
                else:
                    # i-th qubit = 0
                    if bs[step * i: step * i + step] != zero:
                        raise ValueError("The StateVector doesn't represent a n-qubit")
            ampli[n] = sv[bs]
        norm = np.sqrt(np.sum(abs(ampli) ** 2))
        ampli = ampli / norm

        return Qiskit_sv(ampli)

    def sv_to_perceval(self, q_sv):
        r"""Converts a Statevector from qiskit to a StateVector from perceval
        """
        l_sv = len(q_sv)
        zero, one = self._zero_state, self._one_state
        n = np.log2(l_sv)

        if np.round(n, 10) % 1 != 0:
            raise ValueError("The Statevector doesn't represent a n-qubit: the argument length is not a power of 2")
        n = int(n)

        pcvl_sv = StateVector()
        for i in range(l_sv):
            state_i = StateVector()
            bin_i = bin(i)[2:]
            bin_i = '0' * (n - len(bin_i)) + bin_i

            for bit in bin_i:
                if bit == '0':
                    state_i = state_i * zero
                else:
                    state_i = state_i * one

            pcvl_sv += q_sv[i] * state_i

        return pcvl_sv
