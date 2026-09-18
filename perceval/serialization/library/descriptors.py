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

from abc import abstractmethod, ABC
import base64
from typing import TypeAlias, TypeVar

# from typing import Self  # TODO: python 3.11

from .string_buffer import StringBuffer


JsonType = TypeVar("JsonType")  # Any JSON serializable type


class ADescriptor(ABC):

    @abstractmethod
    def to_txt(self) -> str:
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def from_txt(s: StringBuffer):
        raise NotImplementedError()

    def to_json(self) -> JsonType:
        """Can return any *json* serializable type."""
        return self.to_txt()

    @classmethod
    def from_json(cls, s: JsonType):
        """Directly takes the return value of `to_json`"""
        return cls.from_txt(StringBuffer(s))


class DescriptorNone(ADescriptor):
    def __init__(self, v: None):
        # both arg and self.value are needed for this to be usable in SerializerBasicType
        self.value = None

    def to_txt(self) -> str:
        return ""

    @staticmethod
    def from_txt(s: StringBuffer) -> "DescriptorNone":
        s.get_n(0)
        return DescriptorNone(None)

    def to_json(self) -> None:
        return None

    @classmethod
    def from_json(cls, _) -> "DescriptorNone":
        return DescriptorNone(None)


class DescriptorString(ADescriptor):
    def __init__(self, s: str):
        self.value = s

    def to_txt(self) -> str:
        return f"{len(self.value)} {self.value}"

    @staticmethod
    def from_txt(s: StringBuffer) -> "DescriptorString":
        size = s.get_int()
        return DescriptorString(s.get_n(size))

    def to_json(self) -> str:
        return self.value

    @classmethod
    def from_json(cls, s: str) -> "DescriptorString":
        return DescriptorString(s)


class DescriptorBinary(ADescriptor):
    def __init__(self, b: bytes):
        self.value = b

    def _encode(self) -> str:
        return base64.b64encode(self.value).decode("utf-8")

    @classmethod
    def _decode(cls, value) -> "DescriptorBinary":
        return cls(base64.b64decode(value))

    def to_txt(self) -> str:
        s = self._encode()
        return f"{len(s)} {s}"

    @classmethod
    def from_txt(cls, s: StringBuffer) -> "DescriptorBinary":
        size = s.get_int()
        return cls._decode(s.get_n(size))

    def to_json(self) -> str:
        return self._encode()

    @classmethod
    def from_json(cls, s: str) -> "DescriptorBinary":
        return cls._decode(s)


class DescriptorBool(ADescriptor):
    def __init__(self, b: bool):
        self.value = b

    def to_txt(self) -> str:
        return 'T' if self.value else 'F'

    @staticmethod
    def from_txt(s: StringBuffer) -> "DescriptorBool":
        b = s.get_n(1)
        if b == 'T':
            return DescriptorBool(True)
        elif b == 'F':
            return DescriptorBool(False)
        raise RuntimeError(f"invalid boolean token '{b}'")

    def to_json(self) -> bool:
        return self.value

    @classmethod
    def from_json(cls, s: bool) -> "DescriptorBool":
        return DescriptorBool(s)


class DescriptorInteger(ADescriptor):
    def __init__(self, i: int):
        self.value = i

    def to_txt(self) -> str:
        return str(self.value)

    @staticmethod
    def from_txt(s: StringBuffer) -> "DescriptorInteger":
        return DescriptorInteger(int(s.get_next()))

    def to_json(self) -> int:
        return self.value

    @classmethod
    def from_json(cls, s: int) -> "DescriptorInteger":
        return DescriptorInteger(s)


class DescriptorFloat(ADescriptor):
    def __init__(self, f: float):
        self.value = f

    def to_txt(self) -> str:
        #TODO more detailed format?
        return str(self.value)

    @staticmethod
    def from_txt(s: StringBuffer) -> "DescriptorFloat":
        return DescriptorFloat(float(s.get_next()))

    def to_json(self) -> float:
        return self.value

    @classmethod
    def from_json(cls, s: float) -> "DescriptorFloat":
        return DescriptorFloat(s)


class DescriptorComplex(ADescriptor):
    def __init__(self, c: complex):
        self.value = c

    def to_txt(self) -> str:
        return str(f"{self.value.real} {self.value.imag}")

    @staticmethod
    def from_txt(s: StringBuffer) -> "DescriptorComplex":
        return DescriptorComplex(complex(float(s.get_next()), float(s.get_next())))

    def to_json(self) -> list[float]:
        return [self.value.real, self.value.imag]

    @classmethod
    def from_json(cls, values: list[float]) -> "DescriptorComplex":
        return DescriptorComplex(complex(values[0], values[1]))


class DescriptorList(ADescriptor):
    def __init__(self, l: list[int]):
        self.value = l

    def to_txt(self) -> str:
        res = str(len(self.value))
        for index in self.value:
            res += f" {str(index)}"
        return res

    @staticmethod
    def from_txt(s: StringBuffer) -> "DescriptorList":
        size = s.get_int()
        return DescriptorList( [ s.get_int() for _ in range(size) ] )

    def to_json(self) -> list[int]:
        return self.value

    @classmethod
    def from_json(cls, s: list[int]) -> "DescriptorList":
        return DescriptorList(s)


class DescriptorDict(ADescriptor):
    def __init__(self, d: dict[int, int]):
        self.value = d

    def to_txt(self) -> str:
        res = str(len(self.value))
        for key, value in self.value.items():
            res += f" {str(key)} {str(value)}"
        return res

    @staticmethod
    def from_txt(s: StringBuffer) -> "DescriptorDict":
        size = s.get_int()
        return DescriptorDict({s.get_int(): s.get_int() for _ in range(size)} )

    def to_json(self) -> dict[str, int]:
        return {str(key): value for key, value in self.value.items()}  # Json always consider keys as str

    @classmethod
    def from_json(cls, s: dict[str, int]) -> "DescriptorDict":
        return DescriptorDict({int(key): value for key, value in s.items()} )


class DescriptorClass(ADescriptor):
    def __init__(self, version: int, members: list[tuple[str, int]]):
        self.value = (version, members)

    def to_txt(self) -> str:
        version, members = self.value
        res = f"{version} {len(members)}"
        for name, value in members:
            res += f" {name} {str(value)}"
        return res

    @staticmethod
    def from_txt(s: StringBuffer) -> "DescriptorClass":
        version = s.get_int()
        size = s.get_int()
        return DescriptorClass(version, [ (s.get_next(), s.get_int()) for _ in range(size) ])

    def to_json(self) -> tuple[int, list[tuple[str, int]]]:
        return self.value

    @classmethod
    def from_json(cls, s: tuple[int, list[tuple[str, int]]]) -> "DescriptorClass":
        version, members = s
        return DescriptorClass( version, members )


# (Descriptor of the class, [items it depends on])
PartialRecord: TypeAlias = tuple[ADescriptor, list[object]]
