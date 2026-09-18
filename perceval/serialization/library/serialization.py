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

from typing import Type

from .archive import InputArchive, OutputArchive
from .class_registry import ClassRegistry
from .serializers import ClassWriter, DataReader, ClassReader, DescriptorType, ASerializer, SerializerNone, \
    SerializerBool, create_data_serializer, create_data_split_serializer, create_custom_class_serializer, \
    SerializerInt, SerializerFloat, SerializerComplex, SerializerStr, SerializerList, SerializerDict, \
    SerializerTuple, T, SerializerType, SerializerSet, SerializerBytes


class Serialization:

    @staticmethod
    def serialize(obj, ar: OutputArchive):
        """Adds obj to the output archive, marking it as a root object"""
        ar.add(obj)
        ar.roots.append(ar.get_index(obj))

    @staticmethod
    def deserialize(ar: InputArchive):
        """Deserializes the next root object in the input archive, and returns it

        :raise RuntimeError: if all the roots have been deserialized."""
        if not ar.roots:
            raise RuntimeError("No object in Archive")
        return ar.create(ar.roots.pop(0))


    @staticmethod
    def register(serdes: ASerializer):
        """Method to register a serializer to be able to serialize/deserialize the class it points to"""
        ClassRegistry.register(serdes)

    @staticmethod
    def register_data(cls: type, members: list[str] | None = None, *, version: int = None, tag: str = None) -> None:
        a = ClassRegistry.create_data_serializer(cls, members, version=version, tag=tag)
        ClassRegistry.register(a)

    @staticmethod
    def register_data_split(
                cls: Type[T],
                class_serial_members_write: ClassWriter | None = None,
                class_serial_members_read: DataReader | None = None,
                *,
                version: int = None,
                tag: str = None
            ) -> None:
        ClassRegistry.register(ClassRegistry.create_data_split_serializer(cls, class_serial_members_write, class_serial_members_read, version=version, tag=tag))

    @staticmethod
    def register_custom_class(
                cls: Type[T],
                class_write_custom: ClassWriter | None = None,
                class_read_custom: ClassReader | None = None,
                descriptor_type: DescriptorType | None = None,
                *,
                version: int = None,
                tag: str = None
            ) -> None:
        ClassRegistry.register(ClassRegistry.create_custom_class_serializer(cls, class_write_custom, class_read_custom, descriptor_type, version=version, tag=tag))

    @staticmethod
    def register_class(*args, **kwargs) -> None:
        """
        Entry point for registering a class to be able to serialize/deserialize it.
        Chooses and uses one of the forms ``data``, ``data_split``, and ``custom_class``
        depending on what is inside the class and the given arguments """
        errors = [f"Cannot create serializer for class {args[0]}"]
        try:
            Serialization.register_data(*args, **kwargs)
            return
        except Exception as e:
            errors.append(str(e))
        try:
            Serialization.register_data_split(*args, **kwargs)
            return
        except Exception as e:
            errors.append(str(e))
        try:
            Serialization.register_custom_class(*args, **kwargs)
            return
        except Exception as e:
            errors.append(str(e))

        # raise RuntimeError("Cannot create serializer")
        raise RuntimeError('\n'.join(errors))  # python 3.11: ExceptionGroup ?

# injected to avoid circular dependency
ClassRegistry.create_data_serializer = create_data_serializer
ClassRegistry.create_data_split_serializer = create_data_split_serializer
ClassRegistry.create_custom_class_serializer = create_custom_class_serializer

ClassRegistry.register(SerializerNone())
ClassRegistry.register(SerializerBool())
ClassRegistry.register(SerializerInt())
ClassRegistry.register(SerializerFloat())
ClassRegistry.register(SerializerComplex())
ClassRegistry.register(SerializerStr())
ClassRegistry.register(SerializerBytes())
ClassRegistry.register(SerializerList())
ClassRegistry.register(SerializerDict())
ClassRegistry.register(SerializerTuple())
ClassRegistry.register(SerializerType())
ClassRegistry.register(SerializerSet())
