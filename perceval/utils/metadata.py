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

import re
from importlib.metadata import metadata


class PMetadata():
    _NAME = "perceval"
    _PACKAGE_NAME = "perceval-quandela"
    _METADATA = metadata(_PACKAGE_NAME)
    _REGEX = re.compile(r"(\d+\.\d+(?:\.\d+)*)")

    @staticmethod
    def short_version() -> str:
        return PMetadata._REGEX.findall(PMetadata._METADATA["version"])[0]

    @staticmethod
    def version() -> str:
        return PMetadata._METADATA["version"]

    @staticmethod
    def package_name() -> str:
        return PMetadata._PACKAGE_NAME

    @staticmethod
    def author() -> str:
        return PMetadata._METADATA["author"]

    @staticmethod
    def name() -> str:
        return PMetadata._NAME
