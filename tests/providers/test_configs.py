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

from perceval.utils import ContextManager
from perceval.providers import RemoteConfig, ScalewayConfig, KipuConfig, AbstractRemoteConfig


def config_manager(config: AbstractRemoteConfig, token: str, url: str, proxies = None):
    initial_token = config.get_token()
    initial_url = config.get_url()
    initial_proxies = config.get_proxies()

    def set_values(t, u, p):
        config.set_token(t)
        config.set_url(u)
        config.set_proxies(p)
        config.save()

    return ContextManager(lambda: set_values(token, url, proxies),
                          lambda: set_values(initial_token, initial_url, initial_proxies))


def test_no_overlap_and_save():
    q_conf = RemoteConfig()
    s_conf = ScalewayConfig()
    k_conf = KipuConfig()

    with config_manager(q_conf, "q", "q_url"):
        with config_manager(s_conf, "s", "s_url"):
            with config_manager(k_conf, "k", "k_url"):
                assert q_conf.get_token() == "q"
                assert q_conf.get_url() == "q_url"
                assert s_conf.get_token() == "s"
                assert s_conf.get_url() == "s_url"
                assert k_conf.get_token() == "k"
                assert k_conf.get_url() == "k_url"

                # Test save
                assert RemoteConfig().get_token() == "q"
                assert RemoteConfig().get_url() == "q_url"
                assert ScalewayConfig().get_token() == "s"
                assert ScalewayConfig().get_url() == "s_url"
                assert KipuConfig().get_token() == "k"
                assert KipuConfig().get_url() == "k_url"


def test_proxies():
    q_conf = RemoteConfig()
    with config_manager(q_conf, "q", "q_url", {"http": "proxy"}):
        assert q_conf.get_proxies() == {"http": "proxy"}
        assert RemoteConfig().get_proxies() == {"http": "proxy"}

        # Proxies are shared between configs
        assert ScalewayConfig().get_proxies() == {"http": "proxy"}
        assert KipuConfig().get_proxies() == {"http": "proxy"}
