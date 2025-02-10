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
from abc import ABC, abstractmethod
from numbers import Number

from perceval.components import Processor, Circuit, BS, PS, PERM, Port
from perceval.utils import Encoding, P
from perceval.components.component_catalog import CatalogItem

_COMPONENT_STRING_REPR = [
    ("                ╭", "╮"),
    ("Q (dual rail) ──┤", "├── Q (dual rail)"),
    ("              ──┤", "├──"),
    ("                ╰", "╯")]


def _get_component_str_repr(repr_name: str) -> str:
    str_len = len(repr_name) + 2
    output_str = ""
    for i in range(len(_COMPONENT_STRING_REPR)):
        output_str += _COMPONENT_STRING_REPR[i][0]
        if i == 1:
            output_str += f" {repr_name} "
        elif i == 2:
            output_str += ' ' * str_len
        else:
            output_str += '─' * str_len
        output_str += _COMPONENT_STRING_REPR[i][1] + '\n'
    return output_str


class ASingleQubitGate(CatalogItem, ABC):
    repr_name: str
    catalog_name: str
    doc_name: str

    def __init__(self):
        super().__init__(self.catalog_name)

    @property
    def description(self):
        return f"2 modes LO circuit equivalent of the {self.doc_name} single Qubit Gate."

    @property
    def str_repr(self):
        return _get_component_str_repr(self.repr_name)

    def build_processor(self, **kwargs) -> Processor:
        p = self._init_processor(**kwargs)
        return p.add_port(0, Port(Encoding.DUAL_RAIL, 'data'))


class AFixedItem(ASingleQubitGate, ABC):
    circuit: Circuit

    def build_circuit(self, **kwargs) -> Circuit:
        return Circuit(2, name=self.repr_name) // self.circuit


class HadamardItem(AFixedItem):
    repr_name = "H"
    catalog_name = "h"
    doc_name = "Hadamard"
    circuit = BS.H()


class SGateItem(AFixedItem):
    repr_name = "S"
    catalog_name = "s"
    doc_name = repr_name
    circuit = (1, PS(math.pi / 2))


class SDagGateItem(AFixedItem):
    repr_name = "S†"
    catalog_name = "sdag"
    doc_name = ":math:`S^†`"
    circuit = (1, PS(-math.pi / 2))


class TGateItem(AFixedItem):
    repr_name = "T"
    catalog_name = "t"
    doc_name = repr_name
    circuit = (1, PS(math.pi / 4))


class TDagGateItem(AFixedItem):
    repr_name = "T†"
    catalog_name = "tdag"
    doc_name = ":math:`T^†`"
    circuit = (1, PS(-math.pi / 4))


class APauliItem(AFixedItem, ABC):
    axis: str

    @property
    def repr_name(self):
        return self.axis

    @property
    def catalog_name(self):
        return self.axis.lower()

    @property
    def doc_name(self):
        return f"Pauli {self.axis}"


class PauliXItem(APauliItem):
    axis = "X"
    circuit = PERM([1, 0])


class PauliYItem(APauliItem):
    axis = "Y"
    circuit = PERM([1, 0]) // (1, PS(math.pi / 2)) // (0, PS(-math.pi / 2))


class PauliZItem(APauliItem):
    axis = "Z"
    circuit = (1, PS(math.pi))


class AParamItem(ASingleQubitGate, ABC):
    param_key: str

    @abstractmethod
    def get_circuit(self, param) -> Circuit:
        pass

    def build_circuit(self, **kwargs):
        param = kwargs.get(self.param_key, 0.0)
        param = self._handle_param(param)
        name = self.repr_name
        if isinstance(param, Number):
            name = f"{self.repr_name}({param:.3})"
        elif isinstance(param, P):
            name = f"{self.repr_name}({param.name})"
        return Circuit(2, name) // self.get_circuit(param)


class PhaseShiftItem(AParamItem):
    repr_name = "Ps"
    catalog_name = "ph"
    doc_name = "Phase Shifter"
    param_key = "phi"

    def get_circuit(self, phi):
        return Circuit(2)//(1, PS(phi))


class ARItem(AParamItem, ABC):
    axis: str
    param_key = "theta"

    @property
    def repr_name(self):
        return f"R{self.axis.lower()}"

    @property
    def catalog_name(self):
        return self.repr_name.lower()

    @property
    def doc_name(self):
        return self.repr_name


class RxItem(ARItem):
    axis = "X"

    def get_circuit(self, theta):
        return BS.Rx(theta=-theta)


class RyItem(ARItem):
    axis = "Y"

    def get_circuit(self, theta):
        return BS.Ry(theta=theta)


class RzItem(ARItem):
    axis = "Z"

    def get_circuit(self, theta):
        return Circuit(2) // (0, PS(-theta / 2)) // (1, PS(theta / 2))
