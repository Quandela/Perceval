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
import re
import pytest
import platform

from perceval.utils import PersistentData


def test_directory():
    pd = PersistentData()
    if platform.system() == "Linux":
        # '/home/my_user/.local/share/perceval-quandela'
        assert re.match(r"^\/home\/.+\/\.local\/share\/perceval-quandela$", pd.directory) is not None
    elif platform.system() == "Windows":
        # 'C:\\Users\\my_user\\AppData\\Local\\quandela\\perceval-quandela'
        assert re.match(r"^C:\\Users\\.+\\AppData\\Local\\quandela\\perceval-quandela$", pd.directory) is not None
    elif platform.system() == "Darwin":
        # '/Users/my_user/Library/Application Support/perceval-quandela'
        assert re.match(r"^\/Users\/.+\/Library\/Application Support\/perceval-quandela$", pd.directory) is not None
    else:
        raise OSError("My god where are you ?")


def test_basic_methods():
    persistent_data = PersistentData()
    assert os.path.exists(persistent_data.directory)
    assert persistent_data.is_writable()
    assert persistent_data.is_readable()

    persistent_data.create_file("toto")
    assert os.path.exists(os.path.join(persistent_data.directory, "toto"))
    with pytest.warns(UserWarning):
        persistent_data.create_file("toto")

    persistent_data.delete_file("toto")
    assert not os.path.exists(os.path.join(persistent_data.directory, "toto"))
    with pytest.warns(UserWarning):
        persistent_data.delete_file("toto")
    with pytest.raises(FileNotFoundError):
        persistent_data.read_binary_file("toto")

    persistent_data.write_binary_file("toto", b"DEADBEEFDEADBEEF")
    assert persistent_data.read_binary_file("toto") == b"DEADBEEFDEADBEEF"
    persistent_data.delete_file("toto")


def test_access():
    persistent_data = PersistentData()
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

    parent_directory = os.path.dirname(directory)

    os.chmod(parent_directory, 0o000)
    with pytest.warns(UserWarning):
        PersistentData()

    os.chmod(parent_directory, 0o444)
    with pytest.warns(UserWarning):
        PersistentData()

    os.chmod(parent_directory, 0o777)
