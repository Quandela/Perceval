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

from perceval import FockState, NoiseModel
from perceval.serialization import serialize
from perceval.serialization._serialized_containers import SerializedList, SerializedDict

def test_basic_dict():
    # No Perceval object, no nested containers

    d = {1: "str", "str": 6.67}

    d = SerializedDict(d)

    assert 1 in d
    assert d[1] == "str"

    assert "str" in d
    assert d["str"] == 6.67

    assert d.get("unknown") is None

    # Includes perceval object
    fs = FockState([1, 2])
    nm = NoiseModel(0.4)

    d = SerializedDict({0: fs})

    assert d[0] == serialize(fs)

    d = SerializedDict({fs: 1})

    assert fs in d
    assert d[fs] == 1
    assert serialize(fs) in d  # This is a side effect of storing directly the serialized versions of objects

    d.update({fs: 2})
    assert d[fs] == 2

    d["noise"] = nm
    assert d["noise"] == serialize(nm)

    # Test jsonifiable
    json.dumps(d)


def test_basic_list():
    # No Perceval object, no nested containers

    l = SerializedList([1, 2, "str"])

    assert l[0] == 1
    assert l[2] == "str"

    # Includes perceval object
    fs = FockState([1, 2])
    nm = NoiseModel(0.4)

    l = SerializedList([1, fs, nm])

    assert l[1] == serialize(fs)
    assert l[2] == serialize(nm)

    l += [fs]

    assert l[-1] == serialize(fs)
    assert l.count(fs) == 2

    l2 = l + [nm]
    assert l2[-1] == serialize(nm)

    # Test jsonifiable
    json.dumps(l)


def test_nested_containers():
    fs = FockState([1, 2])
    nm = NoiseModel(0.4)

    l = SerializedList([1, 2, {fs: 0}])
    assert isinstance(l[2], SerializedDict)

    d = SerializedDict({fs: [nm]})
    assert isinstance(d[fs], SerializedList)
