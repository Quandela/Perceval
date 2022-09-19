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
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from abc import ABC, abstractmethod
from enum import Enum
from typing import Type

from perceval.backends import Backend, BACKEND_LIST
from .remote_backend import RemoteBackend

DEFAULT_URL = "https://api.cloud.quandela.dev"


class PlatformType(Enum):
    UNKNOWN = 0
    SIMULATOR = 1
    PHYSICAL = 2


class Platform(ABC):
    def __init__(self, name):
        self._name = name
        self._type = PlatformType.UNKNOWN

    @property
    def name(self):
        return self._name

    @property
    def type(self):
        return self._type

    @abstractmethod
    def is_remote(self) -> bool:
        pass

    def is_available(self) -> bool:
        return True

    @abstractmethod
    def backend(self, *args, **kwargs) -> Type[Backend]:
        pass


class LocalPlatform(Platform):
    def __init__(self, name):
        super().__init__(name)
        self._type = PlatformType.SIMULATOR
        self._params = {}

    def set_parameter(self, param_name, value):
        self._params[param_name] = value

    def clear_parameters(self):
        self._params = {}

    def backend(self, *args, **kwargs):
        return BACKEND_LIST[self._name](*args, **kwargs)

    def is_remote(self) -> bool:
        return False


class RemotePlatform(Platform):
    def __init__(self, name: str, token: str, endpoint: str, type: PlatformType):
        super().__init__(name)
        self._token = token
        self._endpoint_prefix = endpoint
        self._type = type

    def build_endpoint(self, suffix: str) -> str:
        return self._endpoint_prefix + suffix

    def is_remote(self) -> bool:
        return True

    def backend(self, *args, **kwargs):
        return RemoteBackend(self, *args, **kwargs)

    def is_available(self) -> bool:
        # TODO request platform availability
        return True  # Temp

    def get_http_headers(self):
        return {'Authorization': f"Bearer {self._token}"}


def _platform_type(endpoint: str, token: str) -> PlatformType:
    # TODO request platform type to the server via a dedicated web service
    return PlatformType.SIMULATOR  # Temp


def get_platform(name: str, token: str = None, url: str = None):
    name = name.lower()
    if url is None:
        url = DEFAULT_URL

    if token is not None:
        return RemotePlatform(name, token, url, _platform_type(url, token))

    if name in BACKEND_LIST:
        return LocalPlatform(name)
    else:
        raise RuntimeError(f"Platform '{name}' not found")
