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

import os
import platform
import pytest

from perceval.utils import PersistentData, FileFormat

from _test_utils import PersistentDataForTests


def test_directory():
    persistent_data = PersistentData()
    assert os.path.isabs(persistent_data.directory)
    assert persistent_data.directory.endswith("perceval-quandela")


def test_basic_methods():
    persistent_data = PersistentDataForTests()
    persistent_data.clear_all_data()

    assert os.path.exists(persistent_data.directory)
    assert persistent_data.is_writable()
    assert persistent_data.is_readable()

    assert not persistent_data.has_file("toto")
    persistent_data.write_file("toto", b"", FileFormat.BINARY)
    assert os.path.exists(os.path.join(persistent_data.directory, "toto"))
    assert persistent_data.has_file("toto")

    persistent_data.delete_file("toto")
    assert not persistent_data.has_file("toto")
    with pytest.warns(UserWarning):
        persistent_data.delete_file("toto")
    with pytest.raises(FileNotFoundError):
        persistent_data.read_file("toto", FileFormat.BINARY)

    persistent_data.write_file("toto", b"DEADBEEFDEADBEEF", FileFormat.BINARY)
    assert persistent_data.read_file("toto", FileFormat.BINARY) == b"DEADBEEFDEADBEEF"

    assert persistent_data.get_folder_size() == 16
    persistent_data.delete_file("toto")
    assert persistent_data.get_folder_size() == 0

    persistent_data.write_file("toto", "DEADBEEFDEADBEEF", FileFormat.TEXT)
    assert persistent_data.read_file("toto", FileFormat.TEXT) == "DEADBEEFDEADBEEF"

    assert persistent_data.get_folder_size() == 16
    persistent_data.delete_file("toto")
    assert persistent_data.get_folder_size() == 0

    with pytest.raises(TypeError):
        persistent_data.write_file("toto", "DEADBEEFDEADBEEF", FileFormat.BINARY)
    persistent_data.delete_file("toto")
    with pytest.raises(TypeError):
        persistent_data.write_file("toto", b"DEADBEEFDEADBEEF", FileFormat.TEXT)

    persistent_data.clear_all_data()


@pytest.mark.skipif(platform.system() == "Windows", reason="chmod doesn't works on windows")
def test_access():
    persistent_data = PersistentDataForTests()
    directory = persistent_data.directory

    os.chmod(directory, 0o000)
    assert not persistent_data.is_writable()
    assert not persistent_data.is_readable()

    os.chmod(directory, 0o444)
    assert not persistent_data.is_writable()
    assert persistent_data.is_readable()

    os.chmod(directory, 0o777)
    assert persistent_data.is_writable()
    assert persistent_data.is_readable()

    persistent_data.clear_all_data()
