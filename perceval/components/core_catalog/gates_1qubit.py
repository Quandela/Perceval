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
    # TODO : keep the following handleparam until i have all gates implemented and tested
    # params_doc = {
    #     'phi_a': "first phase of the MZI (default 'phi_a')",
    #     'phi_b': "second phase of the MZI (default 'phi_b')",
    #     'theta_a': "theta value of the first beam splitter (default pi/2)",
    #     'theta_b': "theta value of the second beam splitter (default pi/2)",
    # }
    #
    # @staticmethod
    # def _handle_params(**kwargs):
    #     if "i" in kwargs:
    #         kwargs["phi_a"] = f"phi_a{kwargs['i']}"
    #         kwargs["phi_b"] = f"phi_b{kwargs['i']}"
    #     return CatalogItem._handle_param(kwargs.get("phi_a", "phi_a")), \
    #         CatalogItem._handle_param(kwargs.get("phi_b", "phi_b")), \
    #         CatalogItem._handle_param(kwargs.get("theta_a", math.pi/2)), \
    #         CatalogItem._handle_param(kwargs.get("theta_b", math.pi/2))

    def build_processor(self, **kwargs) -> Processor:
        return self._init_processor(**kwargs)

    def generate(self, i: int):
        return self.build_circuit(i=i)


class HadamardItem(ASingleQubitGate):
    str_repr = r"""    ╭─────╮╭─────╮╭─────╮╭─────╮
0:──┤phi_a├┤BS.Rx├┤phi_b├┤BS.Rx├──:0
    ╰─────╯│     │╰─────╯│     │
1:─────────┤     ├───────┤     ├──:1
           ╰─────╯       ╰─────╯ """


    def __init__(self):
        super().__init__('Hadamard')

    def build_circuit(self, **kwargs) -> Circuit:
        return Circuit(2, name="H") // BS.H()


class PauliXItem(ASingleQubitGate):
    str_repr = r"""    ╭─────╮╭─────╮╭─────╮╭─────╮
0:──┤phi_a├┤BS.Rx├┤phi_b├┤BS.Rx├──:0
    ╰─────╯│     │╰─────╯│     │
1:─────────┤     ├───────┤     ├──:1
           ╰─────╯       ╰─────╯ """


    def __init__(self):
        super().__init__('Pauli X')

    def build_circuit(self, **kwargs) -> Circuit:
        return Circuit(2, name='X') // PERM([1,0])


# TODO : one class for rotation X gates? with params?
class Rx90Item(ASingleQubitGate):
    str_repr = r"""    ╭─────╮╭─────╮╭─────╮╭─────╮
0:──┤phi_a├┤BS.Rx├┤phi_b├┤BS.Rx├──:0
    ╰─────╯│     │╰─────╯│     │
1:─────────┤     ├───────┤     ├──:1
           ╰─────╯       ╰─────╯ """


    def __init__(self):
        super().__init__('Rotation X90')

    def build_circuit(self, **kwargs) -> Circuit:
        return Circuit(2, name="Rx(π / 2)") // BS.Rx(theta=-math.pi / 2)


class Rxm90Item(ASingleQubitGate):
    str_repr = r"""    ╭─────╮╭─────╮╭─────╮╭─────╮
0:──┤phi_a├┤BS.Rx├┤phi_b├┤BS.Rx├──:0
    ╰─────╯│     │╰─────╯│     │
1:─────────┤     ├───────┤     ├──:1
           ╰─────╯       ╰─────╯ """


    def __init__(self):
        super().__init__('Rotation mX90')

    def build_circuit(self, **kwargs) -> Circuit:
        return Circuit(2, name="Rx(-π / 2)") // BS.Rx(theta=math.pi / 2)


class RxItem(ASingleQubitGate):
    str_repr = r"""    ╭─────╮╭─────╮╭─────╮╭─────╮
0:──┤phi_a├┤BS.Rx├┤phi_b├┤BS.Rx├──:0
    ╰─────╯│     │╰─────╯│     │
1:─────────┤     ├───────┤     ├──:1
           ╰─────╯       ╰─────╯ """


    def __init__(self):
        super().__init__('Rotation X')

    def build_circuit(self, **kwargs) -> Circuit:
        theta = self._handle_params(**kwargs)
        return Circuit(2, name=f"Rx({theta:.3})") // BS.Rx(theta=-theta)

class PauliYItem(ASingleQubitGate):
    str_repr = r"""    ╭─────╮╭─────╮╭─────╮╭─────╮
0:──┤phi_a├┤BS.Rx├┤phi_b├┤BS.Rx├──:0
    ╰─────╯│     │╰─────╯│     │
1:─────────┤     ├───────┤     ├──:1
           ╰─────╯       ╰─────╯ """


    def __init__(self):
        super().__init__('Pauli Y')

    def build_circuit(self, **kwargs) -> Circuit:
        return Circuit(2, name="Y") // PERM([1, 0]) // (1, PS(math.pi / 2)) // (0, PS(-math.pi / 2))


# TODO : one class for rotation X gates? with params?
class Ry90Item(ASingleQubitGate):
    str_repr = r"""    ╭─────╮╭─────╮╭─────╮╭─────╮
0:──┤phi_a├┤BS.Rx├┤phi_b├┤BS.Rx├──:0
    ╰─────╯│     │╰─────╯│     │
1:─────────┤     ├───────┤     ├──:1
           ╰─────╯       ╰─────╯ """


    def __init__(self):
        super().__init__('Rotation Y90')

    def build_circuit(self, **kwargs) -> Circuit:
        return Circuit(2, name="Ry(π / 2)") // BS.Ry(theta=math.pi / 2)


class Rym90Item(ASingleQubitGate):
    str_repr = r"""    ╭─────╮╭─────╮╭─────╮╭─────╮
0:──┤phi_a├┤BS.Rx├┤phi_b├┤BS.Rx├──:0
    ╰─────╯│     │╰─────╯│     │
1:─────────┤     ├───────┤     ├──:1
           ╰─────╯       ╰─────╯ """


    def __init__(self):
        super().__init__('Rotation mY90')

    def build_circuit(self, **kwargs) -> Circuit:
        return Circuit(2, name="Ry(-π / 2)") // BS.Ry(theta=-math.pi / 2)


class RyItem(ASingleQubitGate):
    str_repr = r"""    ╭─────╮╭─────╮╭─────╮╭─────╮
0:──┤phi_a├┤BS.Rx├┤phi_b├┤BS.Rx├──:0
    ╰─────╯│     │╰─────╯│     │
1:─────────┤     ├───────┤     ├──:1
           ╰─────╯       ╰─────╯ """


    def __init__(self):
        super().__init__('Rotation Y')

    def build_circuit(self, **kwargs) -> Circuit:
        theta = self._handle_params(**kwargs)
        return Circuit(2, name=f"Ry({theta:.3})") // BS.Ry(theta=theta)


class PauliZItem(ASingleQubitGate):
    str_repr = r"""    ╭─────╮╭─────╮╭─────╮
0:──┤BS.Rx├┤phi_a├┤BS.Rx├──:0
    │     │╰─────╯│     │
    │     │╭─────╮│     │
1:──┤     ├┤phi_b├┤     ├──:1
    ╰─────╯╰─────╯╰─────╯ """

    def __init__(self):
        super().__init__('Pauli Z')

    def build_circuit(self, **kwargs) -> Circuit:
        return Circuit(2, name="Z") // (1, PS(-math.pi))


class RzItem(ASingleQubitGate):
    str_repr = r"""    ╭─────╮╭─────╮╭─────╮╭─────╮
0:──┤phi_a├┤BS.Rx├┤phi_b├┤BS.Rx├──:0
    ╰─────╯│     │╰─────╯│     │
1:─────────┤     ├───────┤     ├──:1
           ╰─────╯       ╰─────╯ """


    def __init__(self):
        super().__init__('Rotation Z')

    def build_circuit(self, **kwargs) -> Circuit:
        theta = self._handle_params(**kwargs)
        return Circuit(2, name="Rz({theta:.3})") // (0, PS(-theta / 2)) // (1, PS(theta / 2))


# TODO: All T, Tdag, S, Sdag are PHASE gates - maybe we combine them together.
#  although they exist as separate names in both qiskit and cqasm, maybe we use paramterization here?
class SItem(ASingleQubitGate):
    str_repr = r"""    ╭─────╮       ╭─────╮
0:──┤BS.Rx├───────┤BS.Rx├─────────:0
    │     │╭─────╮│     │╭─────╮
1:──┤     ├┤phi_a├┤     ├┤phi_b├──:1
    ╰─────╯╰─────╯╰─────╯╰─────╯ """

    def __init__(self):
        super().__init__('S gate')

    def build_circuit(self, **kwargs) -> Circuit:
        return Circuit(2, name="S") // (1, PS(math.pi / 2))


class SDagItem(ASingleQubitGate):
    str_repr = r"""    ╭─────╮       ╭─────╮
0:──┤BS.Rx├───────┤BS.Rx├─────────:0
    │     │╭─────╮│     │╭─────╮
1:──┤     ├┤phi_a├┤     ├┤phi_b├──:1
    ╰─────╯╰─────╯╰─────╯╰─────╯ """

    def __init__(self):
        super().__init__('S gate')

    def build_circuit(self, **kwargs) -> Circuit:
        return Circuit(2, name="Sdag") // (1, PS(-math.pi / 2))


class TransposeGateItem(ASingleQubitGate):
    str_repr = r"""    ╭─────╮       ╭─────╮
0:──┤BS.Rx├───────┤BS.Rx├─────────:0
    │     │╭─────╮│     │╭─────╮
1:──┤     ├┤phi_a├┤     ├┤phi_b├──:1
    ╰─────╯╰─────╯╰─────╯╰─────╯ """

    def __init__(self):
        super().__init__('T gate')

    def build_circuit(self, **kwargs) -> Circuit:
        return Circuit(2, name="T") // (1, PS(math.pi / 4))


class TDagGateItem(ASingleQubitGate):
    str_repr = r"""    ╭─────╮       ╭─────╮
0:──┤BS.Rx├───────┤BS.Rx├─────────:0
    │     │╭─────╮│     │╭─────╮
1:──┤     ├┤phi_a├┤     ├┤phi_b├──:1
    ╰─────╯╰─────╯╰─────╯╰─────╯ """

    def __init__(self):
        super().__init__('Tdag gate')

    def build_circuit(self, **kwargs) -> Circuit:
        return Circuit(2, name="Tdag") // (1, PS(-math.pi / 4))
