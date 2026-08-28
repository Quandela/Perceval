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

class StringBuffer:
    def __init__(self, value: str):
        self.s = value
        self.current = 0

    def __bool__(self):
        return self.current < len(self.s)

    def get_next(self) -> str:
        start = self.current
        end = self.s.find(' ', start)
        if end == -1: end = len(self.s)
        self.current = end + 1
        return self.s[start:end]

    def get_until_next_token(self, token: str) -> str:
        start = self.current
        end = self.s.find(token, start)
        if end == -1: raise RuntimeError(f"Missing token '{token}' in archive")
        end += len(token)
        self.current = end + 1
        return self.s[start:end]

    def get_n(self, n: int) -> str:
        start = self.current
        end = start + n
        self.current = end + 1
        return self.s[start:end]

    def get_int(self) -> int:
        return int(self.get_next())
