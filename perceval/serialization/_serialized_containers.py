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
import sys

from .serialize import serialize

from multipledispatch import dispatch


class SerializedDict(dict):
    """
    Class that mimics a python dict, but internally stores serialized versions of the perceval objects given to it,
    so that json.dumps can always be called on this class (assuming this only contains python native and perceval objects)

    The compression parameter is always the default one. If another value is wanted, call :code:`serialize`
    with the wanted compression parameter before setting the key or value.

    Retrieving the value of a key will always return the serialized value.
    """

    def __init__(self, seq=None, **kwargs):
        d = dict(seq, **kwargs)

        for key, value in d.items():
            self[key] = value

    def __setitem__(self, key, value):
        return super().__setitem__(serialize(key), make_serialized(value))

    def __delitem__(self, key):
        return super().__delitem__(serialize(key))

    def __getitem__(self, item):
        return super().__getitem__(serialize(item))

    def get(self, key, default=None):
        return super().get(serialize(key), default)

    def pop(self, key, default=None):
        return super().pop(serialize(key), default)

    @staticmethod
    def fromkeys(*args, **kwargs):
        return SerializedDict(dict.fromkeys(*args, **kwargs))

    def __contains__(self, item):
        return super().__contains__(serialize(item))

    def update(self, E=None, **F):
        if E:
            if hasattr(E, 'keys'):
                for key in E.keys():
                    self[key] = E[key]
            else:
                for key, value in E:
                    self[key] = value

        for key, value in F.items():
            self[key] = value

    def __ior__(self, other):
        self.update(**other)
        return self

    def __or__(self, other):
        temp = SerializedDict(self)
        temp |= other
        return temp


class SerializedList(list):
    """
    Class that mimics a python list, but internally stores serialized versions of the perceval objects given to it,
    so that json.dumps can always be called on this class (assuming this only contains python native and perceval objects)

    The compression parameter is always the default one. If another value is wanted, call :code:`serialize`
    with the wanted compression parameter before setting the key or value.

    Retrieving the value of a key will always return the serialized value.
    """

    def __init__(self, seq=()):
        for value in seq:
            self.append(value)

    def __setitem__(self, key, value):
        return super().__setitem__(key, make_serialized(value))

    def append(self, value):
        return super().append(make_serialized(value))

    def count(self, __value):
        return super().count(serialize(__value))

    def extend(self, __iterable):
        return super().extend(SerializedList(__iterable))

    def index(self, __value, __start = 0, __stop = sys.maxsize):
        return super().index(serialize(__value), __start, __stop)

    def insert(self, __index, value):
        return super().insert(__index, make_serialized(value))

    def remove(self, value):
        return super().remove(serialize(value))

    def __add__(self, other):
        return super().__add__(SerializedList(other))

    def __iadd__(self, other):
        return super().__iadd__(SerializedList(other))

    def __contains__(self, item):
        return super().__contains__(serialize(item))

    def __delitem__(self, key):
        return super().__delitem__(serialize(key))


@dispatch(dict)
def make_serialized(d: dict):
    if isinstance(d, SerializedDict):
        return d  # We want to avoid a copy
    return SerializedDict(d)

@dispatch(list)
def make_serialized(l: list):
    if isinstance(l, SerializedList):
        return l
    return SerializedList(l)

@dispatch(object)
def make_serialized(o: object):
    return serialize(o)
