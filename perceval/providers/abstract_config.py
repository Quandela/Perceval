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

from perceval.utils.persistent_data import PersistentData

PROXIES_KEY = "proxies"
URL_KEY = "url"
TOKEN_KEY = "token"


class AbstractRemoteConfig:
    """Handle the remote configuration.

    :param persistent_data: The persistent data access to use. In a standard environment, always use the default.
    """
    # Variables to be set by the subclass
    _token_env_var: str
    _DEFAULT_URL: str
    _REMOTE_KEY: str = "remote"  # Shared for proxies

    _proxies = None  # Shared among all subclasses
    _token = None
    _url = None

    _FIELDS = {
        URL_KEY: "_url",
        TOKEN_KEY: "_token"
    }

    def __init__(self, persistent_data: PersistentData = PersistentData()): # !! default value is evaluated during file load, so test_remote_config_env_var_vs_cache cannot mock it !!
        # TODO: there might be problems on autoloading if we allow the PersistentData not to be the default one
        self._persistent_data = persistent_data

    def _get_remote_config(self, key, remote_key: str = None) -> str | dict[str, str] | None:
        config = self._persistent_data.load_config()
        remote_key = remote_key or self._REMOTE_KEY
        if remote_key in config:
            return config[remote_key].get(key)
        return None

    @classmethod
    def _get_token_from_env_var(cls) -> str | None:
        return os.getenv(cls._token_env_var)

    @staticmethod
    def set_proxies(proxies: dict[str, str]) -> None:
        """
        Set the proxy configuration. The proxy configuration is shared between all configurations.

        Usage example:

        >>> rc = RemoteConfig()
        >>> rc.set_proxies({"http": "http://user:pass@192.168.0.1",
        ...                 "https": "http://user:pass@192.168.0.1:8080"
        ...                })

        :param proxies: proxy configuration in the form of a dictionary which maps protocols to URLs
        """
        AbstractRemoteConfig._proxies = proxies

    def get_proxies(self) -> dict[str, str]:
        """Get the proxy configuration as a mapping of protocols to URLs."""
        if not self._proxies:
            self.set_proxies(self._get_remote_config(PROXIES_KEY, AbstractRemoteConfig._REMOTE_KEY))
        return self._proxies or {}

    @classmethod
    def set_url(cls, url: str) -> None:
        """Set a cloud URL in the configuration cache. It is not saved on disk before the `save` method
        is called.

        :param url: The cloud URL
        """
        cls._url = url

    def get_url(self) -> str:
        """Search a valid cloud URL from the environment, put it in cache and return it.

        The priority for the URL search is as follows:
        * A URL already in cache (e.g. set by the user or already found in a previous call)
        * The value in Perceval persistent configuration

        :return: The cloud URL
        """
        if not self._url:
            self.set_url(self._get_remote_config(URL_KEY))
        return self._url or self._DEFAULT_URL

    @classmethod
    def set_token(cls, token: str) -> None:
        """Set a user authentication token in the configuration cache. It is not saved on disk before the `save` method
        is called.

        :param token: The token
        """
        cls._token = token

    def get_token(self) -> str:
        f"""Search a valid token from the environment, put it in cache and return it.

        The priority for the token search is as follows:
        * A token already in cache (e.g. set by the user or already found in a previous call)
        * The value of the environment variable given by self.get_token_env_var()
        * The value in Perceval persistent configuration

        :return: The token
        """
        if not self._token:
            self.set_token(self._get_token_from_env_var() or self._get_remote_config(TOKEN_KEY))
        return self._token or ""

    @classmethod
    def set_token_env_var(cls, env_var: str) -> None:
        f"""Change the name of the environment variable storing a token.

        :param env_var: name of the new environment variable to search for
        """
        cls._token_env_var = env_var
        # reload the token
        new_token = cls._get_token_from_env_var()
        if new_token:
            cls._token = new_token

    @classmethod
    def get_token_env_var(cls) -> str:
        """Get the name of the environment variable storing a token."""
        return cls._token_env_var

    @classmethod
    def clear_cache(cls):
        """Delete the RemoteConfig cache."""
        AbstractRemoteConfig._proxies = None
        for field, attribute in cls._FIELDS.items():
            setattr(cls, attribute, None)

    def save(self) -> None:
        """Save the current remote configuration on disk.
        After this, the configuration is persistent and can be found in other Perceval sessions (even in different
        virtual envs)."""
        cls = type(self)
        config = self._persistent_data.load_config()
        if self._REMOTE_KEY not in config:
            config[self._REMOTE_KEY] = {}
        if AbstractRemoteConfig._REMOTE_KEY not in config:
            config[AbstractRemoteConfig._REMOTE_KEY] = {}

        config[AbstractRemoteConfig._REMOTE_KEY][PROXIES_KEY] = self._proxies
        for field, attribute in cls._FIELDS.items():
            config[self._REMOTE_KEY][field] = getattr(cls, attribute)

        self._persistent_data.save_config(config)
