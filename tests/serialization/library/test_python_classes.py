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

from math import sqrt

from perceval.serialization import OutputArchive, Serialization, InputArchive


def test_basic_objects():
    # Test write
    ar = OutputArchive()

    a1 = 12
    Serialization.serialize(a1, ar)
    assert ar.memo_decoded() == [ ('int', 12) ]
    assert ar.roots == [ 0 ]
    assert ar.to_text() == "pcvlar 0 1 0 int 12"

    a2 = None
    Serialization.serialize(a2, ar)
    assert ar.memo_decoded() == [ ('int', 12), ('None', None) ]
    assert ar.roots == [ 0, 1 ]
    assert ar.to_text() == "pcvlar 0 2 0 1 int 12 None "

    a3 = "Hello"
    Serialization.serialize(a3, ar)
    assert ar.memo_decoded() == [ ('int', 12), ('None', None), ('str', "Hello") ]
    assert ar.roots == [ 0, 1, 2 ]
    assert ar.to_text() == "pcvlar 0 3 0 1 2 int 12 None  str 5 Hello"

    a4 = 3.14
    Serialization.serialize(a4, ar)
    assert ar.memo_decoded() == [ ('int', 12), ('None', None), ('str', "Hello"), ('float', 3.14) ]
    assert ar.roots == [ 0, 1, 2, 3 ]
    assert ar.to_text() == "pcvlar 0 4 0 1 2 3 int 12 None  str 5 Hello float 3.14"

    a5 = complex(1, sqrt(3))/2
    Serialization.serialize(a5, ar)
    assert ar.memo_decoded() == [ ('int', 12), ('None', None), ('str', "Hello"), ('float', 3.14), ('complex', (0.5+0.8660254037844386j)) ]
    assert ar.roots == [ 0, 1, 2, 3, 4 ]
    assert ar.to_text() == "pcvlar 0 5 0 1 2 3 4 int 12 None  str 5 Hello float 3.14 complex 0.5 0.8660254037844386"

    a6 = b"0x42"
    Serialization.serialize(a6, ar)
    assert ar.memo_decoded() == [ ('int', 12), ('None', None), ('str', "Hello"), ('float', 3.14), ('complex', (0.5+0.8660254037844386j)), ('bytes', b"0x42") ]
    assert ar.roots == [ 0, 1, 2, 3, 4, 5 ]
    assert ar.to_text() == "pcvlar 0 6 0 1 2 3 4 5 int 12 None  str 5 Hello float 3.14 complex 0.5 0.8660254037844386 bytes 8 MHg0Mg=="

    # Test read
    deser_ar = InputArchive.from_text(ar.to_text())
    assert ar.memo_decoded() == deser_ar.memo_decoded()
    assert ar.roots == deser_ar.roots

    assert Serialization.deserialize(deser_ar) == a1
    assert Serialization.deserialize(deser_ar) == a2
    assert Serialization.deserialize(deser_ar) == a3
    assert Serialization.deserialize(deser_ar) == a4
    assert Serialization.deserialize(deser_ar) == a5
    assert Serialization.deserialize(deser_ar) == a6


def test_python_list():
    ar = OutputArchive()
    l = [12, 32]
    Serialization.serialize(l, ar)
    assert ar.memo_decoded() == [ ('list', [1, 2]), ('int', 12), ('int', 32) ]
    assert ar.roots == [ 0 ]
    assert ar.to_text() == "pcvlar 0 1 0 list 2 1 2 int 12 int 32"

    # Test read
    deser_ar = InputArchive.from_text(ar.to_text())
    assert ar.memo_decoded() == deser_ar.memo_decoded()
    assert ar.roots == deser_ar.roots

    assert Serialization.deserialize(deser_ar) == l


def test_python_tuple():
    ar = OutputArchive()
    l = (12, 32)
    Serialization.serialize(l, ar)
    assert ar.memo_decoded() == [ ('tuple', [1, 2]), ('int', 12), ('int', 32) ]
    assert ar.roots == [ 0 ]
    assert ar.to_text() == "pcvlar 0 1 0 tuple 2 1 2 int 12 int 32"

    # Test read
    deser_ar = InputArchive.from_text(ar.to_text())
    assert ar.memo_decoded() == deser_ar.memo_decoded()
    assert ar.roots == deser_ar.roots

    assert Serialization.deserialize(deser_ar) == l


def test_python_dict():
    ar = OutputArchive()
    l = { 'a': 12, 1: 32 }
    Serialization.serialize(l, ar)
    assert ar.memo_decoded() == [ ('dict', [1, 2, 3, 4]), ('str', "a"), ('int', 1), ('int', 12), ('int', 32) ]
    assert ar.roots == [ 0 ]
    assert ar.to_text() == "pcvlar 0 1 0 dict 4 1 2 3 4 str 1 a int 1 int 12 int 32"

    # Test read
    deser_ar = InputArchive.from_text(ar.to_text())
    assert ar.memo_decoded() == deser_ar.memo_decoded()
    assert ar.roots == deser_ar.roots

    assert Serialization.deserialize(deser_ar) == l


def test_python_set():
    ar = OutputArchive()
    l = {"12", 32}
    Serialization.serialize(l, ar)
    assert ar.roots == [0]
    if ar.memo_decoded() == [('set', [1, 2]), ('str', "12"), ('int', 32)]:  # Order is unknown due to set properties
        assert ar.to_text() == "pcvlar 0 1 0 set 2 1 2 str 2 12 int 32"
    elif ar.memo_decoded() == [('set', [1, 2]), ('int', 32), ('str', "12")]:
        assert ar.to_text() == "pcvlar 0 1 0 set 2 1 2 int 32 str 2 12"
    else:
        raise AssertionError(f"{ar.memo_decoded()} != [('set', [1, 2]), ('str', '12'), ('int', 32)]")

    # Test read
    deser_ar = InputArchive.from_text(ar.to_text())
    assert ar.memo_decoded() == deser_ar.memo_decoded()
    assert ar.roots == deser_ar.roots

    assert Serialization.deserialize(deser_ar) == l


def test_python_type():
    ar = OutputArchive()
    l = int
    Serialization.serialize(l, ar)
    assert ar.memo_decoded() == [ ('type', "int") ]
    assert ar.roots == [ 0 ]
    assert ar.to_text() == "pcvlar 0 1 0 type 3 int"

    # Test read
    deser_ar = InputArchive.from_text(ar.to_text())
    assert ar.memo_decoded() == deser_ar.memo_decoded()
    assert ar.roots == deser_ar.roots

    assert Serialization.deserialize(deser_ar) == l


def test_repeated_objects():
    ar = OutputArchive()
    a = [12]
    l = [a, a]
    Serialization.serialize(l, ar)
    assert ar.memo_decoded() == [('list', [1, 1]), ('list', [2]), ('int', 12)]
    assert ar.roots == [0]
    assert ar.to_text() == "pcvlar 0 1 0 list 2 1 1 list 1 2 int 12"

    # Test read
    deser_ar = InputArchive.from_text(ar.to_text())
    assert ar.memo_decoded() == deser_ar.memo_decoded()
    assert ar.roots == deser_ar.roots

    deser = Serialization.deserialize(deser_ar)
    assert deser == l
    assert deser[0] is deser[1]


def test_nested_objects():
    ar = OutputArchive()
    a = []
    l = [a]
    a.append(l)

    Serialization.serialize(l, ar)
    assert ar.memo_decoded() == [('list', [1]), ('list', [0])]
    assert ar.roots == [0]
    assert ar.to_text() == "pcvlar 0 1 0 list 1 1 list 1 0"

    # Test read
    deser_ar = InputArchive.from_text(ar.to_text())
    assert ar.memo_decoded() == deser_ar.memo_decoded()
    assert ar.roots == deser_ar.roots

    deser = Serialization.deserialize(deser_ar)
    assert deser[0][0] is deser


def test_compression():
    ar = OutputArchive()
    l = 12

    Serialization.serialize(l, ar)

    # Test read
    deser_ar = InputArchive.from_text(ar.to_text(compress=True))
    deser = Serialization.deserialize(deser_ar)
    assert deser == l
