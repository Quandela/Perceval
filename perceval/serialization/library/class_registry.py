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

from typing import Callable, Any


class ClassRegistry:
    _serializers_by_class: dict[type, Any] = {}
    _serializers_by_tag: dict[str, Any] = {}

    # Filled later in serialization.py due to circular imports
    create_data_serializer: Callable
    create_data_split_serializer: Callable
    create_custom_class_serializer: Callable

    @staticmethod
    def register(serdes):
        if serdes.type in ClassRegistry._serializers_by_class:
            raise RuntimeError(f"type {serdes.type.__name__} already registered")
        if serdes.class_tag in ClassRegistry._serializers_by_tag:
            raise RuntimeError(f"tag {serdes.class_tag} already registered")
        ClassRegistry._serializers_by_class[serdes.type] = serdes
        ClassRegistry._serializers_by_tag[serdes.class_tag] = serdes

    @staticmethod
    def get_by_class(cls: type):
        if cls in ClassRegistry._serializers_by_class:
            return ClassRegistry._serializers_by_class[cls]
        raise RuntimeError(f"Serializers: unhandled class '{cls.__name__}'")

    @staticmethod
    def get_by_tag(tag: str):
        if tag in ClassRegistry._serializers_by_tag:
            return ClassRegistry._serializers_by_tag[tag]
        raise RuntimeError (f"Serializers: unhandled tag '{tag}'")
