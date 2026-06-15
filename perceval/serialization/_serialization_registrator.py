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

import json
from typing import Any, Callable, TypeVar, Type

from ._type_deserialization import add_type_deserializer
from .serialize import serialize, _handle_compress_parameter, _handle_compression
from .deserialize import deserialize, DESERIALIZER
from ._constants import PCVL_PREFIX, SEP


def default_serializer(obj: Any) -> str:
    # If some attributes should not be serialized, provide a custom serializer/deserializer as they will need to be set back in the returned object
    d = obj.__dict__

    if hasattr(type(obj), "class_version"):
        d["class_version"] = obj.class_version  # Can be used to make retrocompatible deserializers

    return json.dumps(serialize(d))

def default_deserializer(cls, serial: str) -> Any:
    serial_dict: dict = json.loads(serial)
    serial_dict = deserialize(serial_dict)
    serial_version = serial_dict.pop("class_version", 0)

    deserializer = None
    if hasattr(cls, "get_version_deserializer"):
        deserializer = cls.get_version_deserializer(serial_version)

    if deserializer is not None:
        return deserializer(serial_dict)

    # Default deserialization - the version is the same, or pray that the class is still compatible
    obj = cls.__new__(cls)  # Supposes the __new__ method is not overloaded
    obj.__dict__ = serial_dict
    return obj


T = TypeVar("T")

def _register_serializer(cls: Type[T], serialize_method: Callable[[T], str], tag: str, default_compress: bool) -> None:
    def serializer(obj: T, compress=None) -> Any:
        if compress is None:
            compress = default_compress
        compress = _handle_compress_parameter(compress, tag)
        return _handle_compression(f"{PCVL_PREFIX}{tag}{SEP}{serialize_method(obj)}", do_compress=compress)

    serialize.add((cls,), serializer)

def _register_deserializer(deserialize_method : Callable[[str], T], tag: str) -> None:
    DESERIALIZER[tag] = deserialize_method


def register_to_serialization(cls: Type[T],
                              tag: str = None,
                              serialize_method: Callable[[T], str] | None = None,
                              deserialize_method: Callable[[str], T] | None = None,
                              default_compress=False) -> None:
    """
    Adds a class as a valid argument type for the `serialize` and `deserialize` methods.

    By default, the whole __dict__ dictionary is serialized and loaded through recursive calls to `serialize` and json.
    Since this is not very permissive, and subjects to errors when changing a class members, the class may have:
        - "class_version": int class attribute, that is serialized in any instance of the class if present.
        - "get_version_deserializer": Callable[[int], Callable[[dict], cls]] method. Gives a method to call to deserialize an object with another "class_version". It receives the __dict__ attribute of the serialized object

    If the default serialization doesn't correspond to the needs, custom serialize and deserialize methods can be given.

    :param cls: The class to register in serialization.
    :param tag: The tag to use as an identifier for this class. Defaults to the class name.
    :param serialize_method: A method used to produce a string representation of the class instance. By default, serializes as described above
    :param deserialize_method: A method used to produce a class instance from the result of the "serialize_method". By default, deserializes as described above
    :param default_compress: Whether to compress the resulting string by default.
    """
    if tag is None:
        tag = cls.__name__

    if serialize_method is None:
        serialize_method = default_serializer

    _register_serializer(cls, serialize_method, tag, default_compress)

    if deserialize_method is None:
        deserialize_method = lambda serial: default_deserializer(cls, serial)

    _register_deserializer(deserialize_method, tag)
    add_type_deserializer(cls)
