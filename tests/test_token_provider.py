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

from unittest.mock import patch

import perceval as pcvl
from perceval.runtime.remote_config import DEPRECATED_TOKEN_FILENAME
from perceval.utils.persistent_data import _CONFIG_FILE_NAME

from _mock_persistent_data import TokenProviderForTest
from _test_utils import LogChecker

MISSING_KEY = "MISSING_ENV_VAR"
ENV_VAR_KEY = "DUMMY_ENV_VAR"
TOKEN_FROM_ENV = "DUMMY_TOKEN_FROM_ENV"
TOKEN_FROM_CACHE = "DUMMY_TOKEN_FROM_CACHE"
TOKEN_FROM_FILE = "DUMMY_TOKEN_FROM_FILE"


def test_token_provider_env_var_vs_cache(tmp_path):
    os.environ[ENV_VAR_KEY] = TOKEN_FROM_ENV  # Write a temporary environment variable

    provider = TokenProviderForTest(tmp_path, env_var=MISSING_KEY)
    assert provider._from_environment_variable() is None
    assert provider.cache is None

    provider = TokenProviderForTest(tmp_path, env_var=ENV_VAR_KEY)
    assert provider.get_token() == TOKEN_FROM_ENV
    assert provider.cache == TOKEN_FROM_ENV

    provider.force_token(TOKEN_FROM_CACHE)
    assert provider.get_token() == TOKEN_FROM_CACHE

    provider.clear_cache()
    assert provider.cache is None
    assert provider.get_token() == TOKEN_FROM_ENV

    del os.environ[ENV_VAR_KEY]  # Remove the environment variable
    provider.clear_cache()
    assert provider._from_environment_variable() is None


def test_token_provider_from_file(tmp_path):
    token_provider = TokenProviderForTest(tmp_path)
    persistent_data = token_provider._persistent_data
    if persistent_data.load_config():
        pytest.skip("Skipping this test because of an existing user config")
    persistent_data.clear_all_data()

    assert token_provider.get_token() is None
    token_provider.force_token(TOKEN_FROM_FILE)
    token_provider.save_token()
    assert token_provider.get_token() == TOKEN_FROM_FILE

    token_provider.clear_cache()
    persistent_data.clear_all_data()

    assert token_provider.get_token() is None

    token_provider.force_token(TOKEN_FROM_FILE)
    token_provider.save_token()
    assert token_provider.get_token() == TOKEN_FROM_FILE

    token_provider.force_token(TOKEN_FROM_FILE)
    token_provider.save_token()
    assert token_provider.get_token() == TOKEN_FROM_FILE

    token_provider.clear_cache()
    persistent_data.clear_all_data()

    assert token_provider.get_token() is None


@patch.object(pcvl.utils.logging.ExqaliburLogger, "warn")
@pytest.mark.skipif(platform.system() == "Windows", reason="chmod doesn't works on windows")
def test_token_file_access(mock_warn, tmp_path):
    token_provider = TokenProviderForTest(tmp_path)
    persistent_data = token_provider._persistent_data
    if persistent_data.load_config():
        pytest.skip("Skipping this test because of an existing user config")
    directory = persistent_data.directory

    os.chmod(directory, 0o000)

    with pytest.warns(UserWarning) and LogChecker(mock_warn, expected_log_number=1):
        # warnings because config file cannot be saved and TokenProvider is deprecated
        token_provider.force_token(TOKEN_FROM_FILE)
        token_provider.save_token()

    os.chmod(directory, 0o777)

    old_token_file = os.path.join(directory, DEPRECATED_TOKEN_FILENAME)
    new_token_file = os.path.join(directory, _CONFIG_FILE_NAME)
    token_provider.force_token(TOKEN_FROM_FILE)
    token_provider.save_token()

    token_provider.clear_cache()
    assert token_provider.get_token() == TOKEN_FROM_FILE

    os.chmod(old_token_file, 0o000)
    os.chmod(new_token_file, 0o000)

    with pytest.warns(UserWarning) and LogChecker(mock_warn, expected_log_number=4):
        # warnings because config file cannot be retrieved and TokenProvider is deprecated
        temp_token_provider = TokenProviderForTest(tmp_path)
        temp_token_provider._remote_config._persistent_data = persistent_data
        temp_token_provider.clear_cache()
        assert temp_token_provider.get_token() is None

    os.chmod(old_token_file, 0o777)
    os.chmod(new_token_file, 0o777)

    persistent_data.clear_all_data()
