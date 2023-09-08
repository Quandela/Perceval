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
import warnings

from ..utils import PersistentData, FileFormat

_TOKEN_FILE_NAME = "token"


class TokenProvider:
    """Search for a token in different places and retrieve it

    The priority order is:
    - Token cached in memory
    - Environment variable
    - File on the disk
    """

    _CACHED_TOKEN = None

    def __init__(self, env_var: str = "PCVL_CLOUD_TOKEN"):
        """
        :param env_var: Environment variable name to search for a token (default PCVL_CLOUD_TOKEN)
        :param file_path: Path to search for a file containing a token (default None)
        """
        self._env_var = env_var

    def _from_environment_variable(self) -> str:
        if not self._env_var:
            return None
        TokenProvider._CACHED_TOKEN = os.getenv(self._env_var)
        return TokenProvider._CACHED_TOKEN

    def _from_file(self) -> str:
        token = None
        persistent_data = PersistentData()
        if persistent_data.has_file(_TOKEN_FILE_NAME):
            try:
                token = persistent_data.read_file(_TOKEN_FILE_NAME, FileFormat.TEXT)
            except OSError:
                warnings.warn("Cannot read token persistent file")
        return token

    def get_token(self) -> str:
        """Search for a token to provide

        :return: A token, or None if no token was found
        """
        return TokenProvider._CACHED_TOKEN or self._from_environment_variable() or self._from_file()

    @staticmethod
    def clear_cache():
        """Clear the cached token"""
        TokenProvider._CACHED_TOKEN = None

    @property
    def cache(self) -> str:
        return TokenProvider._CACHED_TOKEN

    @staticmethod
    def force_token(token: str):
        """Force a token to be used (and provided to callers)"""
        TokenProvider._CACHED_TOKEN = token


def save_token(token: str):
    """Save provided token into persistent data, replace any token previously saved

    :param token: token to save
    """
    persistent_data = PersistentData()
    if persistent_data.is_writable():
        persistent_data.write_file(_TOKEN_FILE_NAME, token, FileFormat.TEXT)
    else:
        warnings.warn(UserWarning("Can't save token"))
