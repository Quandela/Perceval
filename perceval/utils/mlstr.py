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

import re


def _align(s):
    m = max([len(s) for s in s.split("\n")])
    return "\n".join([ls+" "*(m-len(ls)) for ls in s.split("\n")])


# noinspection PyPep8Naming
class mlstr:
    """
        multiline str utility - useful to build strings with multiline blocks
        should behave mostly as a string except for that specificity
    """
    def __init__(self, *value):
        self._s = str(*value)

    def __iadd__(self, s):
        S = s
        n = len(self._s.split("\n"))
        m = len(S.split("\n"))
        if n > m:
            S += "\n" * (n-m)
        else:
            self._s += "\n" * (m-n)
        self._s = "\n".join([s1+s2 for (s1, s2) in zip(_align(self._s).split("\n"), S.split("\n"))])
        return self

    def __radd__(self, s):
        return mlstr(s)+self

    def __add__(self, s):
        s0 = self._s
        S = str(s)
        n = len(s0.split("\n"))
        m = len(S.split("\n"))
        if n > m:
            S += "\n" * (n-m)
        else:
            s0 += "\n" * (m-n)
        return mlstr("\n".join([s1+s2 for (s1, s2) in zip(_align(s0).split("\n"), S.split("\n"))]))

    def __str__(self):
        return self._s

    def __repr__(self):
        return self._s

    def split(self, sep):
        if sep == "\n":
            return self._s.split(sep)
        else:
            raise NotImplementedError

    @property
    def height(self):
        return len(self._s.split("\n"))

    def __mod__(self, args):
        pats = re.split(r'(%[.0]*[sdf])', self._s)
        s = mlstr('')
        for i, p in enumerate(pats):
            if i % 2:
                s += p % args[i >> 1]
            else:
                s += p
        return s

    def join(self, list_str):
        S = mlstr("")
        for idx, mls in enumerate(list_str):
            if idx:
                S += self._s
            S += mls
        return S
