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
import os

from perceval.utils import FileFormat
from perceval.utils.logging import channel, get_logger
from perceval.utils.persistent_data import PersistentData, _CONFIG_FILE_NAME

QUANDELA_CLOUD_URL = 'https://api.cloud.quandela.com'

REMOTE_KEY = "remote"
PROXIES_KEY = "proxies"
URL_KEY = "url"
TOKEN_KEY = "token"

TOKEN_ENV_VAR = "PCVL_CLOUD_TOKEN"

DEPRECATED_TOKEN_FILENAME = "token"


class RemoteConfig:
    """Handle the remote configuration.

    :param persistent_data: The persistent data access to use. In a standard environment, always use the default.
    """
    _token_env_var = TOKEN_ENV_VAR
    _proxies = None
    _token = None
    _url = None
    _cloud_maximal_job_count = None

    def __init__(self, persistent_data: PersistentData = PersistentData()):
        self._persistent_data = persistent_data

    def _get_deprecated_token(self):
        # check if a token is stored in the deprecated 'token' file
        token = None
        if self._persistent_data.has_file(DEPRECATED_TOKEN_FILENAME):
            get_logger().warn(
                f"The token should now be saved into '{_CONFIG_FILE_NAME}' instead of '{DEPRECATED_TOKEN_FILENAME}'."
                f"Use RemoteConfig class methods `set_token` then `save` to save the token into {_CONFIG_FILE_NAME}."
                f"The file '{DEPRECATED_TOKEN_FILENAME}' is deprecated.", channel.user)
            try:
                token = self._persistent_data.read_file(DEPRECATED_TOKEN_FILENAME, FileFormat.TEXT)
            except OSError:
                get_logger().warn("Cannot read token persistent file", channel.user)
        return token

    def _get_remote_config(self, key) -> str | dict[str, str] | None:
        config = self._persistent_data.load_config()
        if REMOTE_KEY in config:
            return config[REMOTE_KEY].get(key)
        return None

    @staticmethod
    def _get_token_from_env_var() -> str | None:
        return os.getenv(RemoteConfig._token_env_var)

    @staticmethod
    def set_proxies(proxies: dict[str, str]) -> None:
        """
        Set the proxy configuration.

        Usage example:

        >>> rc = RemoteConfig()
        >>> rc.set_proxies({"http": "http://user:pass@192.168.0.1",
        ...                 "https": "http://user:pass@192.168.0.1:8080"
        ...                })

        :param proxies: proxy configuration in the form of a dictionary which maps protocols to URLs
        """
        RemoteConfig._proxies = proxies

    def get_proxies(self) -> dict[str, str]:
        """Get the proxy configuration as a mapping of protocols to URLs."""
        if not RemoteConfig._proxies:
            RemoteConfig._proxies = self._get_remote_config(PROXIES_KEY)
        return RemoteConfig._proxies or {}

    @staticmethod
    def set_url(url: str) -> None:
        """Set a cloud URL in the configuration cache. It is not saved on disk before the `save` method
        is called.

        :param url: The cloud URL
        """
        RemoteConfig._url = url

    def get_url(self) -> str:
        """Search a valid cloud URL from the environment, put it in cache and return it.

        The priority for the URL search is as follows:
        * A URL already in cache (e.g. set by the user or already found in a previous call)
        * The value in Perceval persistent configuration

        :return: The cloud URL
        """
        if not RemoteConfig._url:
            RemoteConfig._url = self._get_remote_config(URL_KEY)
        return RemoteConfig._url or QUANDELA_CLOUD_URL

    @staticmethod
    def set_token(token: str) -> None:
        """Set a user authentication token in the configuration cache. It is not saved on disk before the `save` method
        is called.

        :param token: The token
        """
        RemoteConfig._token = token

    def get_token(self) -> str:
        f"""Search a valid token from the environment, put it in cache and return it.

        The priority for the token search is as follows:
        * A token already in cache (e.g. set by the user or already found in a previous call)
        * The value of the "{TOKEN_ENV_VAR}" environment variable
        * The value in Perceval persistent configuration

        :return: The token
        """
        if not RemoteConfig._token:
            RemoteConfig._token = self._get_token_from_env_var() or self._get_remote_config(TOKEN_KEY) or self._get_deprecated_token()
        return RemoteConfig._token or ""

    @staticmethod
    def set_token_env_var(env_var: str) -> None:
        f"""Change the name of the environment variable storing a token. Default is {TOKEN_ENV_VAR}.

        :param env_var: name of the new environment variable to search for
        """
        RemoteConfig._token_env_var = env_var
        # reload the token
        new_token = RemoteConfig._get_token_from_env_var()
        if new_token:
            RemoteConfig._token = new_token

    @staticmethod
    def get_token_env_var() -> str:
        """Get the name of the environment variable storing a token."""
        return RemoteConfig._token_env_var

    @staticmethod
    def set_cloud_maximal_job_count(count: int) -> None:
        RemoteConfig._cloud_maximal_job_count = count

    @staticmethod
    def get_cloud_maximal_job_count() -> int:
        if not RemoteConfig._cloud_maximal_job_count:
            get_logger().warn("""cloud_maximal_job_count is not set.
            Please use RemoteConfig.set_cloud_maximal_job_count([see user pricing plan])""", channel.user)
        return RemoteConfig._cloud_maximal_job_count

    @staticmethod
    def clear_cache():
        """Delete the RemoteConfig cache."""
        RemoteConfig._proxies = None
        RemoteConfig._url = None
        RemoteConfig._token = None

    def save(self) -> None:
        """Save the current remote configuration on disk.
        After this, the configuration is persistent and can be found in other Perceval sessions (even in different
        virtual envs)."""
        config = self._persistent_data.load_config()
        if REMOTE_KEY not in config:
            config[REMOTE_KEY] = {}

        config[REMOTE_KEY][PROXIES_KEY] = RemoteConfig._proxies
        config[REMOTE_KEY][URL_KEY] = RemoteConfig._url
        config[REMOTE_KEY][TOKEN_KEY] = RemoteConfig._token

        self._persistent_data.save_config(config)
