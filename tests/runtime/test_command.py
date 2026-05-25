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

import pytest

from perceval import Command, CommandFactory


def test_fill():
    signature = [("float", float, True), ("list", list, False)]

    command = Command("test", signature)

    assert command.fill(1.2, [3.14]) == {"float": 1.2, "list": [3.14]}
    assert command.fill(1.2, list=[3.14]) == {"float": 1.2, "list": [3.14]}
    assert command.fill(float=1.2, list=[3.14]) == {"float": 1.2, "list": [3.14]}

    assert command.fill(list=[3.14]) == {"list": [3.14]}
    assert command.fill(1.2) == {"float": 1.2}

    assert command.fill() == {}

    with pytest.raises(TypeError):
        command.fill([3.14])  # Incorrect argument type

    with pytest.raises(TypeError):
        command.fill(1.2, 3.14)  # Incorrect argument type

    with pytest.raises(TypeError):
        command.fill(list=3.14)  # Incorrect argument type

    with pytest.raises(TypeError):
        command.fill(1.2, float=3.14)  # Same argument given twice

    with pytest.raises(TypeError):
        command.fill(1.2, [3.14], 42)  # Too many arguments

    with pytest.raises(TypeError):
        command.fill(unknown=3.14)  # Unknown argument


def test_validate():
    signature = [("mandatory", float, True), ("optional", list, False)]

    command = Command("test", signature)

    command.check({"mandatory": 1.2, "optional": [3.14]})
    command.check({"mandatory": 1.2})

    with pytest.raises(ValueError):
        command.check({"optional": [3.14]})  # Missing mandatory


def test_factory():
    assert CommandFactory.probs.name == "probs"
    assert CommandFactory.sample_count.name == "sample_count"
    assert CommandFactory.samples.name == "samples"
