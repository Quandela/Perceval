import numpy as np
from perceval.utils import StateVector
from perceval.utils import BasicState
from enum import Enum
from qiskit.quantum_info import Statevector as qiskit_sv


class Encoding(Enum):
    DUAL_RAIL = 0
    POLARIZATION = 1
    QUDIT = 2
    TIME = 3
    RAW = 4


def sv_to_qsphere_param(sv, ancilla=None, encoding=Encoding.DUAL_RAIL,
                        polarization_base=(BasicState("|{P:H}>"), BasicState("|{P:V}>"))):
    r"""Convert a StateVector in dual rail encoding to be interpreted by qsphere from qiskit
    ancilla are the mode we are supressing to obtain our multi-qubits state"""

    l_sv = len(sv)
    if l_sv == 0:
        raise ValueError("The StateVector is empty")
    if ancilla is not None:
        sv = remove_ancilla(sv, ancilla)

    zero, one = encoding_to_log(encoding, polarization_base=(BasicState("|{P:H}>"), BasicState("|{P:V}>")))
    step = len(zero)
    print(zero, one)

    l_bs = len(sv[0])
    print(step)
    if l_bs % step != 0:
        raise ValueError("The StateVector doesn't represent a n-qubit")
    else:
        l_n_qbt = l_bs // step

    ampli = np.zeros(2 ** l_n_qbt, dtype=complex)
    for state in sv:
        bs = BasicState(state)
        N = 0
        for i in range(l_n_qbt):
            # check the value of each qubit
            # i-th qubit = 1
            print(bs[step * i: step * i + step])
            if bs[step * i: step * i + step] == one:
                N += 2 ** (l_n_qbt - i - 1)
            else:
                # i-th qubit = 0
                print(bs[step * i: step * i + step], zero)
                if bs[step * i: step * i + step] != zero:
                    raise ValueError("The StateVector doesn't represent a n-qubit")
        ampli[N] = sv[bs]
    norm = np.sqrt(np.sum(abs(ampli) ** 2))
    ampli = ampli / norm

    return qiskit_sv(ampli)


def remove_ancilla(sv, anscilla):
    r"""Removes the auxiliary modes to obtain a proper n-qubits state
    """

    anscilla = np.sort(anscilla)
    new_sv = StateVector()
    for state in sv:
        bs = BasicState(state)
        new_bs = StateVector()
        previous = -1
        for i in range(len(anscilla)):
            new_bs = new_bs * bs[previous + 1:anscilla[i]]
            previous = anscilla[i]
        new_sv += sv[bs] * new_bs

    if len(sv) != len(new_sv):
        raise ValueError(
            "The StateVector doesn't represent a n-qubit: some termes have been supressed while removing ancillas")
    else:
        sv = new_sv
    return sv


def encoding_to_log(encoding, polarization_base=(BasicState("|{P:H}>"), BasicState("|{P:V}>"))):
    assert isinstance(encoding, Encoding), "You need to provide an encoding"

    if encoding == Encoding.RAW:
        zero = BasicState("|0>")
        one = BasicState("|1>")
    elif encoding == Encoding.DUAL_RAIL:
        zero = BasicState("|1,0>")
        one = BasicState("|0,1>")
    elif encoding == Encoding.POLARIZATION:
        if len(polarization_base[0]) != 1 or len(polarization_base[1]) != 1:
            raise ValueError("The BasicStates representing the polarization basis should only contain one mode")
        zero = polarization_base[0]
        one = polarization_base[1]
    else:
        raise ValueError("Only use RAW, DUAL_RAIL or POLARIZATION encoding.")

    return zero, one


a = sv_to_qsphere_param(StateVector("|0,1, 0,1, 1,0>") - StateVector("|1,0, 0,1, 1,0>"))
print(a)
