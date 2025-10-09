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
import pytest
import platform

from perceval.utils.persistent_data import _CONFIG_FILE_NAME

from _mock_persistent_data import RemoteConfigForTest, QUANDELA_CLOUD_URL


MISSING_KEY = "MISSING_ENV_VAR"
ENV_VAR_KEY = "DUMMY_ENV_VAR"
TOKEN_FROM_ENV = "DUMMY_TOKEN_FROM_ENV"
TOKEN_FROM_CACHE = "DUMMY_TOKEN_FROM_CACHE"
TOKEN_FROM_FILE = "DUMMY_TOKEN_FROM_FILE"

PROXY_FROM_CACHE = {'https': 'socks5h://USER:PWD@DUMMY_PROXY_FROM_CACHE:1080/'}
PROXY_FROM_FILE = {'https': 'socks5h://USER:PWD@DUMMY_PROXY_FROM_FILE:1080/'}

URL_FROM_CACHE = "DUMMY_URL_FROM_CACHE"
URL_FROM_FILE = "DUMMY_URL_FROM_FILE"


def test_remote_config_env_var_vs_cache(tmp_path):
    os.environ[ENV_VAR_KEY] = TOKEN_FROM_ENV  # Write a temporary environment variable

    remote_config = RemoteConfigForTest(tmp_path)
    assert remote_config._get_token_from_env_var() is None
    assert remote_config._token is None
    assert remote_config._proxies is None
    assert remote_config._url is None

    remote_config.set_token_env_var(ENV_VAR_KEY)
    assert remote_config.get_token() == TOKEN_FROM_ENV
    assert remote_config._token == TOKEN_FROM_ENV

    remote_config.set_token(TOKEN_FROM_CACHE)
    assert remote_config.get_token() == TOKEN_FROM_CACHE
    remote_config.set_proxies(PROXY_FROM_CACHE)
    assert remote_config.get_proxies() == PROXY_FROM_CACHE
    remote_config.set_url(URL_FROM_CACHE)
    assert remote_config.get_url() == URL_FROM_CACHE

    remote_config.clear_cache()
    assert remote_config._token is None
    assert remote_config.get_token() == TOKEN_FROM_ENV
    assert remote_config._proxies is None
    assert remote_config.get_proxies() == {}
    assert remote_config._url is None
    assert remote_config.get_url() == QUANDELA_CLOUD_URL

    del os.environ[ENV_VAR_KEY]  # Remove the environment variable
    remote_config.clear_cache()
    assert remote_config._get_token_from_env_var() is None


def test_remote_config_from_file(tmp_path):
    remote_config = RemoteConfigForTest(tmp_path)
    persistent_data = remote_config._persistent_data
    if persistent_data.load_config():
        pytest.skip("Skipping this test because of an existing user config")
    persistent_data.clear_all_data()

    assert remote_config.get_token() == ''
    assert remote_config.get_proxies() == {}
    assert remote_config.get_url() == QUANDELA_CLOUD_URL
    remote_config.set_token(TOKEN_FROM_FILE)
    remote_config.set_proxies(PROXY_FROM_FILE)
    remote_config.set_url(URL_FROM_FILE)
    remote_config.save()
    assert remote_config.get_token() == TOKEN_FROM_FILE
    assert remote_config.get_proxies() == PROXY_FROM_FILE
    assert remote_config.get_url() == URL_FROM_FILE

    remote_config.clear_cache()
    persistent_data.clear_all_data()

    assert remote_config.get_token() == ''
    assert remote_config.get_proxies() == {}
    assert remote_config.get_url() == QUANDELA_CLOUD_URL

    remote_config.set_token(TOKEN_FROM_FILE)
    remote_config.set_proxies(PROXY_FROM_FILE)
    remote_config.set_url(URL_FROM_FILE)
    remote_config.save()
    assert remote_config.get_token() == TOKEN_FROM_FILE
    assert remote_config.get_proxies() == PROXY_FROM_FILE
    assert remote_config.get_url() == URL_FROM_FILE

    remote_config.set_token(TOKEN_FROM_FILE)
    remote_config.set_proxies(PROXY_FROM_FILE)
    remote_config.set_url(URL_FROM_FILE)
    remote_config.save()
    assert remote_config.get_token() == TOKEN_FROM_FILE
    assert remote_config.get_proxies() == PROXY_FROM_FILE
    assert remote_config.get_url() == URL_FROM_FILE

    remote_config.clear_cache()
    persistent_data.clear_all_data()

    assert remote_config.get_token() == ''
    assert remote_config.get_proxies() == {}
    assert remote_config.get_url() == QUANDELA_CLOUD_URL


@pytest.mark.skipif(platform.system() == "Windows", reason="chmod doesn't works on windows")
def test_config_file_access(tmp_path):
    remote_config = RemoteConfigForTest(tmp_path)
    persistent_data = remote_config._persistent_data
    if persistent_data.load_config():
        pytest.skip("Skipping this test because of an existing user config")
    directory = persistent_data.directory

    os.chmod(directory, 0o000)

    with pytest.warns(UserWarning):
        # warning because config file cannot be saved
        remote_config.set_token(TOKEN_FROM_FILE)
        remote_config.set_proxies(PROXY_FROM_FILE)
        remote_config.set_url(URL_FROM_FILE)
        remote_config.save()

    os.chmod(directory, 0o777)

    token_file = os.path.join(directory, _CONFIG_FILE_NAME)
    remote_config.set_token(TOKEN_FROM_FILE)
    remote_config.set_proxies(PROXY_FROM_FILE)
    remote_config.set_url(URL_FROM_FILE)
    remote_config.save()

    remote_config.clear_cache()
    assert remote_config.get_token() == TOKEN_FROM_FILE
    assert remote_config.get_proxies() == PROXY_FROM_FILE
    assert remote_config.get_url() == URL_FROM_FILE

    os.chmod(token_file, 0o000)

    with pytest.warns(UserWarning):
        # warning because config file cannot be retrieved
        temp_remote_config = RemoteConfigForTest(tmp_path)
        temp_remote_config._persistent_data = persistent_data
        temp_remote_config.clear_cache()
        assert temp_remote_config.get_token() == ''
        assert temp_remote_config.get_proxies() == {}
        assert temp_remote_config.get_url() == QUANDELA_CLOUD_URL

    os.chmod(token_file, 0o777)

    persistent_data.clear_all_data()
