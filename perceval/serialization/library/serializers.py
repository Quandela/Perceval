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

from types import NoneType
from typing import Type, Generic

from .abstract_serializer import ASerializer, T, DescriptorType, PreRecorder, ClassWriter, DataReader, ClassReader
from .archive import InputArchive, OutputArchive
from .descriptors import PartialRecord, DescriptorClass, DescriptorNone, DescriptorBool, DescriptorInteger, \
    DescriptorFloat, DescriptorComplex, DescriptorString, DescriptorList, DescriptorBinary
from .class_registry import ClassRegistry


#########################
# basic class serializers
#########################


class SerializerBasicType(ASerializer[T, DescriptorType]):
    def write(self, obj: T, ar: OutputArchive) -> PartialRecord:
        return self.descriptor_type(obj), []

    def read(self, ar: InputArchive, desc: DescriptorType, pre_recorder: PreRecorder) -> T:
        return desc.value


class SerializerNone(SerializerBasicType[NoneType, DescriptorNone]):
    type = NoneType
    class_tag = 'None'
    descriptor_type = DescriptorNone


class SerializerBool(SerializerBasicType[bool, DescriptorBool]):
    type = bool
    class_tag = 'bool'
    descriptor_type = DescriptorBool


class SerializerInt(SerializerBasicType[int, DescriptorInteger]):
    type = int
    class_tag = 'int'
    descriptor_type = DescriptorInteger


class SerializerFloat(SerializerBasicType[float, DescriptorFloat]):
    type = float
    class_tag = 'float'
    descriptor_type = DescriptorFloat


class SerializerComplex(SerializerBasicType[complex, DescriptorComplex]):
    type = complex
    class_tag = 'complex'
    descriptor_type = DescriptorComplex


class SerializerStr(SerializerBasicType[str, DescriptorString]):
    type = str
    class_tag = 'str'
    descriptor_type = DescriptorString


class SerializerBytes(SerializerBasicType[bytes, DescriptorBinary]):
    type = bytes
    class_tag = 'bytes'
    descriptor_type = DescriptorBinary


class SerializerList(ASerializer):
    type = list
    class_tag = 'list'
    descriptor_type = DescriptorList

    def write(self, obj: list, ar: OutputArchive) -> PartialRecord:
        ar.pre_record(obj)

        return DescriptorList([ar.get_index(v) for v in obj]), obj

    def read(self, ar: InputArchive, desc: DescriptorList, pre_recorder: PreRecorder) -> list:
        obj = []
        # first record empty list so children can point to it
        pre_recorder(obj)

        for child in desc.value:
            obj.append(ar.create(child))

        return obj


class SerializerDict(ASerializer):
    type = dict
    class_tag = 'dict'
    descriptor_type = DescriptorList

    def write(self, obj: dict, ar: OutputArchive) -> PartialRecord:
        keys_then_values = list(obj.keys()) + list(obj.values())

        ar.pre_record(keys_then_values)

        return DescriptorList([ar.get_index(c) for c in keys_then_values]), keys_then_values

    def read(self, ar: InputArchive, desc: DescriptorList, pre_recorder: PreRecorder) -> dict:
        obj = self.type()  # Allows inheriting from this class, as long as the type acts as a dict
        # first record empty dict so children can point to it
        pre_recorder(obj)

        size = len(desc.value)
        if size % 2 == 1:
            raise RuntimeError(f"total count of keys+values is {size}")
        keys = desc.value[:size//2]
        values = desc.value[size//2:]

        for k, v in zip(keys, values):
            obj[ar.create(k)] = ar.create(v)

        return obj


class SerializerTuple(ASerializer):
    type = tuple
    class_tag = 'tuple'
    descriptor_type = DescriptorList

    def write(self, obj: tuple, ar: OutputArchive) -> PartialRecord:
        # no child can contain reference to obj, we can add them safely
        for c in obj:
            ar.add(c)

        return DescriptorList([ar.get_index(c) for c in obj]), []

    def read(self, ar: InputArchive, desc: DescriptorList, pre_recorder: PreRecorder) -> tuple:
        # no child can contain reference to obj, so no need to pre_record
        return tuple( ar.create(idx) for idx in desc.value )


class SerializerType(ASerializer):
    type = type
    class_tag = 'type'
    descriptor_type = DescriptorString

    def write(self, obj: type, ar: OutputArchive) -> PartialRecord:
        return DescriptorString(ClassRegistry.get_by_class(obj).class_tag), []

    def read(self, ar: InputArchive, desc: DescriptorString, pre_recorder: PreRecorder) -> type:
        return ClassRegistry.get_by_tag(desc.value).type


class SerializerSet(ASerializer):
    type = set
    class_tag = 'set'
    descriptor_type = DescriptorList

    def write(self, obj: set, ar: OutputArchive) -> PartialRecord:
        # Sets can't directly contain themselves, but a member could contain it
        ar.pre_record(obj)

        return DescriptorList([ar.get_index(v) for v in obj]), list(obj)

    def read(self, ar: InputArchive, desc: DescriptorList, pre_recorder: PreRecorder) -> set:
        obj = set()
        # first record empty set so children can point to it
        pre_recorder(obj)

        for child in desc.value:
            obj.add(ar.create(child))

        return obj


############################
# custom classes serializers
############################

class SerializerData(ASerializer[T, DescriptorClass]):
    descriptor_type = DescriptorClass

    def __init__(self, cls: Type[T], tag: str, version: int, members: list[str]):
        self.type = cls
        self.class_tag = tag
        self.version = version
        self.members = members

    def write(self, obj: T, ar: OutputArchive) -> PartialRecord:
        children = [ getattr(obj, m) for m in self.members ]

        ar.pre_record(children)

        return (DescriptorClass(self.version, [(name, ar.get_index(value)) for name, value in zip(self.members, children)]),
                children)

    def read(self, ar: InputArchive, desc: DescriptorClass, pre_recorder: PreRecorder) -> T:
        obj = self.type.__new__(self.type)
        # first record empty object so children can point to it
        pre_recorder(obj)

        for member in desc.value[1]:
            name, idx = member
            setattr(obj, name, ar.create(idx))

        return obj


class SerializerDataSplitFunctions(ASerializer[T, DescriptorClass]):
    descriptor_type = DescriptorClass

    def __init__(
                self,
                cls: Type[T],
                tag: str,
                version: int,
                class_serial_members_write: ClassWriter,
                class_serial_members_read: DataReader
            ):
        self.type = cls
        self.class_serial_members_write = class_serial_members_write
        self.class_serial_members_read = class_serial_members_read
        self.class_tag = tag
        self.class_version = version

    def write(self, obj: T, ar: OutputArchive) -> PartialRecord:
        return self.class_serial_members_write(obj, ar)

    def read(self, ar: InputArchive, desc: DescriptorClass, pre_recorder: PreRecorder) -> T:
        obj = self.type.__new__(self.type)
        # first record empty object so children can point to it
        pre_recorder(obj)

        version, members = desc.value
        self.class_serial_members_read(obj, ar, members, version)

        return obj


class SerializerClass(Generic[T, DescriptorType]):

    def __init__(
                self,
                cls: Type[T],
                tag: str,
                version: int,
                class_write_custom: ClassWriter,
                class_read_custom: ClassReader,
                descriptor_type: DescriptorType
            ):
        self.type = cls
        self._write = class_write_custom
        self._read = class_read_custom
        self.class_tag = tag
        self.class_version = version
        self.descriptor_type = descriptor_type

    def write(self, obj: T, ar: OutputArchive) -> PartialRecord:
        return self._write(obj, ar)

    def read(self, ar: InputArchive, desc: DescriptorClass, pre_recorder: PreRecorder) -> T:
        return self._read(ar, desc, pre_recorder)

#########
# Helpers
#########

def create_data_serializer(cls: Type[T], members: list[str] | None = None, *, version: int = None, tag: str = None):
    if members is None:
        if hasattr(cls, 'class_serial_members'):
            members = getattr(cls, 'class_serial_members')
        else:
            # must not default to (filtered) __dict__ or anything else,
            # because evolution of the class would silently break deserialization
            raise RuntimeError(f"missing serialized member list for class {cls.__name__}")
    if tag is None:
        tag = getattr(cls, 'class_tag') if hasattr(cls, 'class_tag') else cls.__name__
    if version is None:
        version = getattr(cls, 'class_version') if hasattr(cls, 'class_version') else 0
    return SerializerData(cls, tag, version, members)


def create_data_split_serializer(
            cls: Type[T],
            class_serial_members_write: ClassWriter | None = None,
            class_serial_members_read: DataReader | None = None,
            *,
            version: int = None,
            tag: str = None
        ) -> SerializerDataSplitFunctions:
    if class_serial_members_write is None:
        if hasattr(cls, 'class_serial_members_write'):
            class_serial_members_write = getattr(cls, 'class_serial_members_write')
        else:
            raise RuntimeError(f"missing class_serial_members_write function for class {cls.__name__}")
    if class_serial_members_read is None:
        if hasattr(cls, 'class_serial_members_read'):
            class_serial_members_read = getattr(cls, 'class_serial_members_read')
        else:
            raise RuntimeError(f"missing class_serial_members_read function for class {cls.__name__}")
    if tag is None:
        tag = getattr(cls, 'class_tag') if hasattr(cls, 'class_tag') else cls.__name__
    if version is None:
        version = getattr(cls, 'class_version') if hasattr(cls, 'class_version') else 0
    return SerializerDataSplitFunctions(cls, tag, version, class_serial_members_write, class_serial_members_read)


def create_custom_class_serializer(
            cls: Type[T],
            class_write_custom: ClassWriter | None = None,
            class_read_custom: ClassReader | None = None,
            descriptor_type: DescriptorType | None = None,
            *,
            version: int = None,
            tag: str = None
        ):
    if class_write_custom is None:
        if hasattr(cls, 'class_write_custom'):
            class_write_custom = getattr(cls, 'class_write_custom')
        else:
            raise RuntimeError(f"missing class_write_custom function for class {cls.__name__}")
    if class_read_custom is None:
        if hasattr(cls, 'class_read_custom'):
            class_read_custom = getattr(cls, 'class_read_custom')
        else:
            raise RuntimeError(f"missing class_read_custom function for class {cls.__name__}")
    if descriptor_type is None:
        if hasattr(cls, 'class_serializer_type'):
            descriptor_type = getattr(cls, 'class_serializer_type')
        else:
            raise RuntimeError(f"missing class_serializer_type for class {cls.__name__}")
    if tag is None:
        tag = getattr(cls, 'class_tag') if hasattr(cls, 'class_tag') else cls.__name__
    if version is None:
        version = getattr(cls, 'class_version') if hasattr(cls, 'class_version') else 0
    return SerializerClass(cls, tag, version, class_write_custom, class_read_custom, descriptor_type)
