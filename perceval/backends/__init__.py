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


from deprecated import deprecated
from .platforms import LocalPlatform, RemotePlatform
from .template import Backend
from .remote import RemoteBackend, RemoteCredentials, Job


@deprecated(reason='Please use get_platform("local") instead')
class BackendFactory(LocalPlatform):
    pass


def get_platform(name: str, credentials: RemoteCredentials = None):
    if isinstance(name, RemoteCredentials):
        credentials = name
        name = '1bf73239-192e-4b23-809f-007b323826f5'  # 'simulator'

    if name is None or name == "local":
        return LocalPlatform()
    else:
        return RemotePlatform(name, credentials)
