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
from __future__ import annotations

from perceval.utils import FileFormat
from perceval.utils.logging import deprecated, get_logger, channel

from perceval.runtime.remote_config import RemoteConfig, DEPRECATED_TOKEN_FILENAME


class TokenProvider:
    """Search for a token in different places and retrieve it

    The priority order is:
    - Token cached in memory
    - Environment variable
    - File on the disk
    """

    @deprecated(version="0.13.0", reason=f"Use RemoteConfig class instead of TokenProvider class")
    def __init__(self, env_var: str = "PCVL_CLOUD_TOKEN", remote_config: RemoteConfig = RemoteConfig()):
        """
        :param env_var: Environment variable name to search for a token (default PCVL_CLOUD_TOKEN)
        :param file_path: Path to search for a file containing a token (default None)
        """
        self._remote_config = remote_config
        self._remote_config.clear_cache()
        self._remote_config.set_token_env_var(env_var)

    @property
    def _persistent_data(self):
        return self._remote_config._persistent_data

    def _from_environment_variable(self) -> str | None:
        return self._remote_config._get_token_from_env_var()

    def get_token(self) -> str | None:
        """Search for a token to provide

        :return: A token, or None if no token was found
        """
        token = self._remote_config.get_token()
        if token != "":
            return token
        return None

    def save_token(self):
        """Save the current cache token
        """
        self._remote_config.save()
        # also save in the old file
        if self._persistent_data.is_writable():
            self._persistent_data.write_file(DEPRECATED_TOKEN_FILENAME, self._remote_config._token, FileFormat.TEXT)
        else:
            get_logger().warn("Can't save token", channel.user)

    @staticmethod
    def clear_cache():
        """Clear the cached token"""
        RemoteConfig.clear_cache()

    @property
    def cache(self) -> str | None:
        return self._remote_config._token

    @staticmethod
    def force_token(token: str):
        """Force a token to be used (and provided to callers)"""
        RemoteConfig.set_token(token)


@deprecated(version="0.13.0", reason="Use instead RemoteConfig class methods `set_token` then `save`")
def save_token(token: str):
    """Save provided token into persistent data, replace any token previously saved

    :param token: token to save
    """
    token_provider = TokenProvider()
    token_provider.force_token(token)
    token_provider.save_token()
