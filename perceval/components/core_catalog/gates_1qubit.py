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

import math
from abc import ABC


from perceval.components import Processor, Circuit, BS, PS, PERM
from perceval.components.component_catalog import CatalogItem

# Todo: all the following gates are from the list in CQASM (unitaryhack)
#  check documentation- qiskit and find good ways of implementing to remain generic
# todo: also check in myqlm ;-)

# todo: fix all str_repr - i have not updated any - all are copied from prvs

# todo: update usage in _pauli.py -> used in tomography, TU,
#  and update usage in converters ofcourse

class ASingleQubitGate(CatalogItem, ABC):
    description = "2 mode LO circuit equivalent of a single Qubit Gate"

    def build_processor(self, **kwargs) -> Processor:
        return self._init_processor(**kwargs)


class HadamardItem(ASingleQubitGate):
    str_repr = r"""                ╭─────╮
Q1:(dual rail)──┤  H  ├── (dual rail) :Q1
              ──┤     ├──
                ╰─────╯ """

    def __init__(self):
        super().__init__('hadamard')

    def build_circuit(self, **kwargs) -> Circuit:
        return Circuit(2, name="H") // BS.H()


class PauliXItem(ASingleQubitGate):
    str_repr = r"""                ╭─────╮
Q1:(dual rail)──┤Pauli├── (dual rail) :Q1
              ──┤  X  ├──
                ╰─────╯ """

    def __init__(self):
        super().__init__('pauli x')

    def build_circuit(self, **kwargs) -> Circuit:
        return Circuit(2, name='X') // PERM([1,0])

#
# class RxPi2Item(ASingleQubitGate):
#     str_repr = r"""    ╭─────╮
# 0:─┤BS.Rx├─:0
#    │     │
# 1:─┤     ├─:1
#    ╰─────╯ """
#
#
#     def __init__(self):
#         super().__init__('rotation x90')
#
#     def build_circuit(self, **kwargs) -> Circuit:
#         return Circuit(2, name="Rx(π / 2)") // BS.Rx(theta=-math.pi / 2)
#
#
# class RxMinusPi2Item(ASingleQubitGate):
#     str_repr = r"""    ╭─────╮
#     0:─┤BS.Rx├─:0
#        │     │
#     1:─┤     ├─:1
#        ╰─────╯ """
#
#     def __init__(self):
#         super().__init__('rotation xminus90')
#
#     def build_circuit(self, **kwargs) -> Circuit:
#         return Circuit(2, name="Rx(-π / 2)") // BS.Rx(theta=math.pi / 2)


class RxItem(ASingleQubitGate):
    str_repr = r"""                ╭─────╮
Q1:(dual rail)──┤ Rx  ├── (dual rail) :Q1
              ──┤     ├──
                ╰─────╯ """

    def __init__(self):
        super().__init__('rx')

    def build_circuit(self, **kwargs) -> Circuit:
        theta = kwargs.get('theta', 0)
        return Circuit(2, name=f"Rx({theta:.3})") // BS.Rx(theta=-theta)

class PauliYItem(ASingleQubitGate):
    str_repr = r"""                ╭─────╮
Q1:(dual rail)──┤Pauli├── (dual rail) :Q1
              ──┤  Y  ├──
                ╰─────╯ """

    def __init__(self):
        super().__init__('pauli y')

    def build_circuit(self, **kwargs) -> Circuit:
        return Circuit(2, name="Y") // PERM([1, 0]) // (1, PS(math.pi / 2)) // (0, PS(-math.pi / 2))


# class Ry90Item(ASingleQubitGate):
#     str_repr = r"""    ╭─────╮
#         0:─┤BS.Ry ├─:0
#            │(π/2) │
#         1:─┤      ├─:1
#            ╰──────╯ """
#
#     def __init__(self):
#         super().__init__('Rotation Y90')
#
#     def build_circuit(self, **kwargs) -> Circuit:
#         return Circuit(2, name="Ry(π / 2)") // BS.Ry(theta=math.pi / 2)
#
#
# class Rym90Item(ASingleQubitGate):
#     str_repr = r"""    ╭──────╮
#     0:─┤BS.Ry ├─:0
#        │(-π/2)│
#     1:─┤      ├─:1
#        ╰──────╯ """
#
#     def __init__(self):
#         super().__init__('Rotation mY90')
#
#     def build_circuit(self, **kwargs) -> Circuit:
#         return Circuit(2, name="Ry(-π / 2)") // BS.Ry(theta=-math.pi / 2)


class RyItem(ASingleQubitGate):
    str_repr = r"""                ╭─────╮
Q1:(dual rail)──┤ Ry  ├── (dual rail) :Q1
              ──┤     ├──
                ╰─────╯ """

    def __init__(self):
        super().__init__('ry')

    def build_circuit(self, **kwargs) -> Circuit:
        theta = kwargs.get('theta', 0)
        return Circuit(2, name=f"Ry({theta:.3})") // BS.Ry(theta=theta)


class PauliZItem(ASingleQubitGate):
    str_repr = r"""                ╭─────╮
Q1:(dual rail)──┤Pauli├── (dual rail) :Q1
              ──┤  Z  ├──
                ╰─────╯ """

    def __init__(self):
        super().__init__('pauli z')

    def build_circuit(self, **kwargs) -> Circuit:
        return Circuit(2, name="Z") // (1, PS(-math.pi))


class RzItem(ASingleQubitGate):
    str_repr = r"""                ╭─────╮
Q1:(dual rail)──┤ Rz  ├── (dual rail) :Q1
              ──┤     ├──
                ╰─────╯ """

    def __init__(self):
        super().__init__('rz')

    def build_circuit(self, **kwargs) -> Circuit:
        theta = kwargs.get('theta', 0)
        return Circuit(2, name=f"Rz({theta:.3})") // (0, PS(-theta / 2)) // (1, PS(theta / 2))


# QISKIT AND CQASM has names s, sdag, t, tdag (also have P -> generic phase shift gate)
# TODO: All T, Tdag, S, Sdag are PHASE gates - maybe we combine them together here and use converter
#  to phase shift values?

class PhaseShiftITem(ASingleQubitGate):
    str_repr = r"""                ╭─────╮
Q1:(dual rail)──┤ PS  ├── (dual rail) :Q1
              ──┤     ├──
                ╰─────╯ """

    def __init__(self):
        super().__init__('phase shift')

    def build_circuit(self, **kwargs) -> Circuit:
        phi = kwargs.get('phi')
        return Circuit(2, name="phase shift") // (1, PS(phi))

class SGateItem(ASingleQubitGate):
    str_repr = r"""                ╭─────╮
Q1:(dual rail)──┤  S  ├── (dual rail) :Q1
              ──┤     ├──
                ╰─────╯ """

    def __init__(self):
        super().__init__('s')

    def build_circuit(self, **kwargs) -> Circuit:
        return Circuit(2, name="S") // (1, PS(math.pi / 2))


class SDagGateItem(ASingleQubitGate):
    str_repr = r"""                ╭─────╮
Q1:(dual rail)──┤ S†  ├── (dual rail) :Q1
              ──┤     ├──
                ╰─────╯ """

    def __init__(self):
        super().__init__('sdag')

    def build_circuit(self, **kwargs) -> Circuit:
        return Circuit(2, name="Sdag") // (1, PS(-math.pi / 2))


class TGateItem(ASingleQubitGate):
    str_repr = r"""                ╭─────╮
Q1:(dual rail)──┤  T  ├── (dual rail) :Q1
              ──┤     ├──
                ╰─────╯ """

    def __init__(self):
        super().__init__('t')

    def build_circuit(self, **kwargs) -> Circuit:
        return Circuit(2, name="T") // (1, PS(math.pi / 4))


class TDagGateItem(ASingleQubitGate):
    str_repr = r"""                ╭─────╮
Q1:(dual rail)──┤ T†  ├── (dual rail) :Q1
              ──┤     ├──
                ╰─────╯ """

    def __init__(self):
        super().__init__('tdag')

    def build_circuit(self, **kwargs) -> Circuit:
        return Circuit(2, name="Tdag") // (1, PS(-math.pi / 4))
