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

from abc import ABC, abstractmethod
from typing import TypeVar, Callable, TypeAlias, Generic

from .archive import OutputArchive, InputArchive
from .descriptors import ADescriptor, PartialRecord

T = TypeVar("T")
DescriptorType = TypeVar("DescriptorType")

PreRecorder: TypeAlias = Callable[[object], None]
ClassWriter: TypeAlias = Callable[[T, OutputArchive], PartialRecord]
DataReader: TypeAlias = Callable[[T, InputArchive, list[tuple[str, int]], int], None]
ClassReader: TypeAlias = Callable[[InputArchive, ADescriptor, PreRecorder], None]


class ASerializer(ABC, Generic[T, DescriptorType]):
    """A class describing how to serialize an object from another class T, by using a descriptor"""
    type: T
    class_tag: str
    descriptor_type: DescriptorType

    @abstractmethod
    def write(self, obj: T, ar: OutputArchive) -> PartialRecord:
        """Makes a Descriptor of the object, and the list of all children that need to be serialized"""
        pass

    @abstractmethod
    def read(self, ar: InputArchive, desc: DescriptorType, pre_recorder: PreRecorder) -> T:
        """Takes a descriptor of the object, and returns a filled instance of the object"""
        pass
