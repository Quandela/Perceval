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

from enum import Enum


class Encoding(Enum):
    """Logical encodings

    QUDITn: a Qudit partition encoding n qubits with 1 photon in 2**n modes
    """
    DUAL_RAIL = 0
    POLARIZATION = 1
    TIME = 3
    RAW = 4
    QUDIT2 = 5
    QUDIT3 = 6
    QUDIT4 = 7
    QUDIT5 = 8
    QUDIT6 = 9
    QUDIT7 = 10  # 2**7 = 128 modes

    @property
    def logical_length(self) -> int:
        """Logical length of an encoding"""
        n = self.name
        if n.startswith("QUDIT"):
            return int(n[len("QUDIT"):])
        return 1

    @property
    def fock_length(self) -> int:
        """Fock state length of an encoding"""
        if self == Encoding.DUAL_RAIL:
            return 2
        elif self == Encoding.POLARIZATION:
            return 1
        elif self == Encoding.TIME:
            return 1
        elif self == Encoding.RAW:
            return 1
        return 2**self.logical_length


class InterferometerShape(Enum):
    RECTANGLE = 0
    TRIANGLE = 1


class FileFormat(Enum):
    BINARY = 0
    TEXT = 1


class ModeType(Enum):
    PHOTONIC = 0
    HERALD = 1
    CLASSICAL = 2
