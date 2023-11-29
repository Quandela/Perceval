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
import shutil
import warnings
from typing import Union
from platformdirs import PlatformDirs

from .metadata import PMetadata
from ._enums import FileFormat



def _removesuffix(data, suffix):
    """Replace the python 3.9 method removesuffix

    :param data: data on which remove suffix
    :param suffix: suffix to remove
    :return: data
    """
    if data.endswith(suffix):
        data = data[:-len(suffix)]

    return data


class PersistentData:
    """PersistentData handle perceval persistent data
    On init, it creates a directory (if it doesn't exist) for storing perceval persistent data
    Directory depends of the os:
    * Linux: '/home/my_user/.local/share/perceval-quandela'
    * Windows: 'C:\\Users\\my_user\\AppData\\Local\\quandela\\perceval-quandela'
    * Darwin: '/Users/my_user/Library/Application Support/perceval-quandela'

    If the directory cannot be created or read/write in, a warning will inform the user
    """

    def __init__(self) -> None:
        self._directory = PlatformDirs(PMetadata.package_name(), PMetadata.author()).user_data_dir
        try:
            self._create_directory()
        except OSError as exc:
            warnings.warn(exc)
            return
        if not self.is_writable() or not self.is_readable():
            warnings.warn(UserWarning(f"Cannot read or write in {self._directory}"))

    def is_writable(self) -> bool:
        """Return if the directory is writable

        :return: True if the directory is writable, False else
        """
        return os.access(self._directory, os.W_OK)

    def is_readable(self) -> bool:
        """Return if the directory is readable

        :return: True if the directory is readable, False else
        """
        return os.access(self._directory, os.R_OK)

    def _create_directory(self) -> None:
        """Create the directory if if doesn't exists
        """
        if not os.path.exists(self._directory):
            os.makedirs(self._directory)

    def get_folder_size(self) -> int:
        """Get the directory data size

        :return: directory data size in bytes
        """
        return sum(os.path.getsize(os.path.join(dirpath, filename))
                   for dirpath, dirnames, filenames in os.walk(self._directory) for filename in filenames)

    def _get_full_path(self, element_name: str) -> str:
        """Get the full path of an element supposedly in persistent data directory

        :param element_name: name of the element (with extension)
        :return: full path of the file
        """
        return os.path.join(self._directory, element_name)

    def has_file(self, filename: str) -> bool:
        """Find if persistant data has file

        :param filename: name of the file to find (with extension)
        :return: True is the file exists, else False
        """
        return os.path.exists(os.path.join(self._directory, filename))

    def delete_file(self, filename: str):
        """Delete a file in persistent data directory
        if file doesn't exist, raise a user warning

        :param filename: name of the file to delete (with extension)
        """
        file_path = self._get_full_path(filename)
        if not os.path.exists(file_path):
            warnings.warn(UserWarning(f"Cannot delete {file_path}, file doesn't exist"))
            return
        os.remove(file_path)

    def write_file(self, filename: str, data: Union[bytes, str], file_format: FileFormat):
        """Write data into a file in persistent data directory

        :param filename: name of the file to write in (with extension)
        :param data: data to write
        """
        file_path = self._get_full_path(filename)

        if file_format == FileFormat.BINARY:
            with open(file_path, "wb") as file:
                file.write(data)
        elif file_format == FileFormat.TEXT:
            with open(file_path, "wt", encoding="UTF-8") as file:
                file.write(data)
        else:
            raise NotImplementedError(f"format {format} is not supported")

    def read_file(self, filename: str, file_format: FileFormat) -> Union[bytes, str]:
        """Read data from a file in persistent data directory

        :param filename: name of the file to read (with extension)
        :raises FileNotFoundError: Raise an exception if file is not found
        :return: data
        """
        file_path = self._get_full_path(filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(file_path)
        data = None

        if file_format == FileFormat.BINARY:
            with open(file_path, "r+b") as file:
                data = file.read()
            data = _removesuffix(data, b'\n')
            data = _removesuffix(data, b' ')
        elif file_format == FileFormat.TEXT:
            with open(file_path, "r+t", encoding="UTF-8") as file:
                data = str(file.read())
            data = _removesuffix(data, '\n').rstrip()
        else:
            raise NotImplementedError(f"format {format} is not supported")
        return data

    def clear_all_data(self):
        """Delete persistent data directory and recreate it
        """
        shutil.rmtree(self._directory)
        self._create_directory()

    @property
    def directory(self) -> str:
        """return persistent data directory

        :return: persistent data directory
        """
        return self._directory
