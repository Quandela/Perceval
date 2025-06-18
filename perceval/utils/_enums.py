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
    """Logical qubit encoding on photons"""
    DUAL_RAIL = 0
    POLARIZATION = 1
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
        elif self == Encoding.RAW:
            return 1
        return 2**self.logical_length

Encoding.DUAL_RAIL.__doc__ = "Dual rail encoding where a qubit is encoded as the position of 1 photon in 2 modes."
Encoding.POLARIZATION.__doc__ = "Qubit is encoding on a single photon polarization (horizontal / vertical) in 1 mode."
Encoding.RAW.__doc__ = "Raw encoding is the closest to photonics. It encodes a qubit as the presence of photons in 1 mode."
Encoding.QUDIT2.__doc__ = ("Qudits are encoding multiple qubits in the location of a single photon in multiple modes. "
                           "QUDIT2 encodes 2 qubits on 1 photon in 4 modes.")
Encoding.QUDIT3.__doc__ = "Encodes 3 qubits on 1 photon in 8 modes."
Encoding.QUDIT4.__doc__ = "Encodes 4 qubits on 1 photon in 16 modes."
Encoding.QUDIT5.__doc__ = "Encodes 5 qubits on 1 photon in 32 modes."
Encoding.QUDIT6.__doc__ = "Encodes 6 qubits on 1 photon in 64 modes."
Encoding.QUDIT7.__doc__ = "Encodes 7 qubits on 1 photon in 128 modes."


class InterferometerShape(Enum):
    RECTANGLE = 0
    TRIANGLE = 1

InterferometerShape.RECTANGLE.__doc__ = ("Rectangular matrix of universal 2-modes components (e.g. MZI). "
                                         "All paths have the same depth.")
InterferometerShape.TRIANGLE.__doc__ = ("Triangular mesh of universal 2-modes components. "
                                        "The top of the interferometer has max depth, the bottom mode has depth 1.")


class FileFormat(Enum):
    BINARY = 0
    TEXT = 1


class ModeType(Enum):
    PHOTONIC = 0
    HERALD = 1
    CLASSICAL = 2

ModeType.PHOTONIC.__doc__ = "Photonic mode. Additional linear optics components can be added on such a mode."
ModeType.HERALD.__doc__ = "Ancillary mode: defines a special photonic mode used for heralding. Nothing can be added."
ModeType.CLASSICAL.__doc__ = ("A classical register represents a measured mode. A `Detector` turns a photonic mode into "
                              "a classical bit. Only classical components can be added.")
