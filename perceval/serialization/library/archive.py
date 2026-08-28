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

import warnings
from typing import Iterable, Any

from .descriptors import DescriptorClass, PartialRecord
from .class_registry import ClassRegistry
from .string_buffer import StringBuffer
from .utils import compress_str, decompress_str


class Archive:

    class NoValue:
        pass

    header = 'pcvlar'
    archive_version = 0

    def __init__(self, raise_on_unregistred_class: bool = True):
        self.raise_on_unregistred_class = raise_on_unregistred_class
        self.roots = [] # list of ids
        self.memo = [] # list of (tag, desc)


class OutputArchive(Archive):
    """
    Archive used to serialize objects
    """
    def __init__(self, raise_on_unregistred_class: bool = True):
        super().__init__(raise_on_unregistred_class)
        self.ids = []

    # for test only
    def memo_decoded(self):
        res = []
        for o in self.memo:
            tag, desc = o
            res += [ (tag, desc.value) ]
        return res

    def get_index(self, obj) -> int:
        i = id(obj)
        return self.ids.index(i)

    def add(self, obj):
        i = id(obj)

        # if i in self.ids:
        if i in self.ids and self.memo[self.ids.index(i)] is not self.NoValue:
            return

        try:
            t = ClassRegistry.get_by_class(type(obj))
        except Exception as e:
            if self.raise_on_unregistred_class:
                raise
            warnings.warn(str(e))
            return

        if i in self.ids:
            idx = self.ids.index(i)
        else:
            idx = len(self.ids)
            self.ids.append(i)
            self.memo.append(self.NoValue)

        record = t.write(obj, self)
        self.memo[idx] = (t.class_tag, record[0])
        for t in record[1]:
            self.add(t)

    def save_attr(self, obj, attributes: list[str]) -> PartialRecord:
        """
        Adds all the given attributes of obj to the archive.

        :param obj: The object that is currently being added to the archive
        :param attributes: A list of attributes of the object that will be added to the archive
        :return: A PartialRecord, i.e. a descriptor and a list of all new objects to add to the archive.
        """
        t = ClassRegistry.get_by_class(type(obj))
        children = [ getattr(obj, m) for m in attributes ]

        self.pre_record(children)

        return (DescriptorClass(t.class_version, [ (name, self.get_index(value)) for name, value in zip(attributes, children) ]),
                children)

    def pre_record(self, children: Iterable[Any]):
        for c in children:
            # Avoid comprehension to prevent repeated objects
            if id(c) not in self.ids:
                self.ids.append(id(c))

        self.memo.extend( [ self.NoValue ] * (len(self.ids) - len(self.memo)) )

    # To storable object
    def to_json(self):
        raise NotImplementedError("JSON storage not implemented")

    def to_text(self, compress: bool = False) -> str:
        """
        :param compress: If True, the resulting string will be compressed
        :return: A string representing all the data stored in the archive
        """
        res = f" {self.archive_version} {len(self.roots)} " + " ".join(map(str, self.roots))
        for entry in self.memo:
            tag, desc = entry
            res += f" {tag} {desc.to_txt()}"
        if compress:
            res = f":zip:{compress_str(res)}"

        res = f"{self.header}" + res
        return res


class InputArchive(Archive):
    """
    Archive used to deserialize objects
    """
    def __init__(self, raise_on_unregistred_class: bool = True):
        super().__init__(raise_on_unregistred_class)
        self.created = []

    # for test only
    def memo_decoded(self):
        res = []
        for o in self.memo:
            tag, desc = o
            res += [ (tag, desc.value) ]
        return res

    def __len__(self) -> int:
        """
        :return: the number of roots remaining to be deserialized
        """
        return len(self.roots)

    def create(self, index: int) -> object:
        """
        :param index: The index of the object to create in the archive
        :return: The object that was created in the archive
        """
        if index < 0 or index >= len(self.memo):
            raise IndexError(f"invalid object index {index} (memo len: {len(self.memo)})")

        if self.created[index] is not self.NoValue:
            return self.created[index]

        tag = self.memo[index][0]
        desc = self.memo[index][1]

        try:
            t = ClassRegistry.get_by_tag(tag)
        except Exception as e:
            if self.raise_on_unregistred_class:
                raise
            warnings.warn(str(e))
            return
        return t.read(self, desc, lambda obj: self.created.__setitem__(index, obj))

    def load_attr(self, obj, desc: list[tuple[str, int]]):
        """
        Creates and loads as attributes all the objects described in `desc` into `obj`.
        :param obj: The object that will receive the attributes.
        :param desc: A list of pairs (attribute_name, index)
        """
        for name, idx in desc:
            setattr(obj, name, self.create(idx))

    # Storable object parsing
    @classmethod
    def from_json(cls):
        raise NotImplementedError("JSON storage not implemented")

    @classmethod
    def from_text(cls, txt: str) -> "InputArchive":  # TODO: python 3.11: use Self
        """
        :param txt: a string representing an archive, typically obtained by using an OutputArchive.to_txt() method.
        :return: A new InputArchive containing the data that were stored in the archive.
        """
        compress_header = f"{InputArchive.header}:zip:"
        if txt.startswith(compress_header):
            txt = f"{InputArchive.header}{decompress_str(txt[len(compress_header):])}"

        self = cls()
        self.roots = []
        self.memo = []
        self.created = []

        buffer = StringBuffer(txt)

        header = buffer.get_next()
        if header != self.header:
            raise RuntimeError(f"invalid archive")

        archive_version = buffer.get_int()
        if archive_version > self.archive_version:
            raise RuntimeError(f"unknown archive version {archive_version}")

        n_roots = buffer.get_int()
        self.roots = [ buffer.get_int() for _ in range(n_roots) ]

        while buffer:
            tag = buffer.get_next()
            t = ClassRegistry.get_by_tag(tag)
            desc = t.descriptor_type.from_txt(buffer)
            self.memo.append( (tag, desc) )

        self.created = [ self.NoValue ] * len(self.memo)
        return self
