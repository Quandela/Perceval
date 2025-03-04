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

from perceval.runtime._token_management import TokenProvider
from perceval.utils.persistent_data import PersistentData

REMOTE_KEY = "remote"
PROXIES_KEY = "proxies"


def _get_proxies():
    config = PersistentData().load_config()
    proxies = {}
    if REMOTE_KEY in config:
        proxies = config[REMOTE_KEY].get(PROXIES_KEY)
    return proxies

def _get_token():
    return TokenProvider().get_token()

class RemoteConfig:
    """Handle the remote configuration"""
    _proxies = _get_proxies()
    _token = _get_token()

    @staticmethod
    def set_proxies(proxies: dict[str,str]) -> None:
        """Set the proxies

        :param proxies: The proxy dict
        """
        RemoteConfig._proxies = proxies

    @staticmethod
    def get_proxies() -> dict[str,str]:
        """Get the current proxies

        :return: Current proxies
        """
        return RemoteConfig._proxies

    @staticmethod
    def set_token(token: str) -> None:
        """Set the token

        :param token: The token
        """
        RemoteConfig._token = token

    @staticmethod
    def get_token() -> str:
        """Get the current token

        :return: Current token
        """
        return RemoteConfig._token

    @staticmethod
    def save():
        """Save the current remote configuration
        """
        token_provider = TokenProvider()
        token_provider.force_token(RemoteConfig._token)
        token_provider.save_token()

        persistent_data = PersistentData()
        config = persistent_data.load_config()
        if REMOTE_KEY not in config:
            config[REMOTE_KEY] = {}
        config[REMOTE_KEY][PROXIES_KEY] = RemoteConfig._proxies
        persistent_data.save_config(config)
