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

import os

MISSING_KEY = "MISSING_ENV_VAR"
ENV_VAR_KEY = "DUMMY_ENV_VAR"
TOKEN_FROM_ENV = "DUMMY_TOKEN_FROM_ENV"
TOKEN_FROM_CACHE = "DUMMY_TOKEN_FROM_CACHE"
os.environ[ENV_VAR_KEY] = TOKEN_FROM_ENV  # Write a temporary environment variable

def test_token_provider_env_var_vs_cache():
    provider = TokenProvider(env_var=MISSING_KEY)
    assert provider.get_token() is None
    assert provider.cache is None

    provider = TokenProvider(env_var=ENV_VAR_KEY)
    assert provider.get_token() == TOKEN_FROM_ENV
    assert provider.cache == TOKEN_FROM_ENV

    provider.force_token(TOKEN_FROM_CACHE)
    assert provider.get_token() == TOKEN_FROM_CACHE

    provider.clear_cache()
    assert provider.cache is None
    assert provider.get_token() == TOKEN_FROM_ENV

    del os.environ[ENV_VAR_KEY]  # Remove the environment variable
    provider.clear_cache()
    assert provider.get_token() is None
