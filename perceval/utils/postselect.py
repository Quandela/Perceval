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

from perceval.utils.statevector import BasicState

import json
import re
from typing import Callable


class PostSelect:
    _OPERATOR = {"==": int.__eq__, "<": int.__lt__, ">": int.__gt__}
    _PATTERN = re.compile(r"(\[[,0-9\s]+\]\s*)(==|<|>)\s*(\d+\b)")

    def __init__(self, str_repr: str = None):
        self._conditions = {}
        condition_count = 0
        if str_repr is not None:

            try:
                for match in self._PATTERN.finditer(str_repr):
                    indexes = tuple(json.loads(match.group(1)))
                    # print(match.group(1), indexes)
                    self._add_condition(indexes=indexes,
                                        operator=self._OPERATOR[match.group(2)],
                                        value=int(match.group(3)))
                    condition_count += 1
            except json.decoder.JSONDecodeError as e:
                raise RuntimeError(f"Could not interpret input string '{str_repr}': {e}")
            if condition_count != str_repr.count("&") + 1:
                raise RuntimeError(f"Could not interpret input string '{str_repr}': Invalid format")

    def eq(self, indexes, value: int):
        self._add_condition(indexes, int.__eq__, value)
        return self

    def gt(self, indexes, value: int):
        self._add_condition(indexes, int.__gt__, value)
        return self

    def lt(self, indexes, value: int):
        self._add_condition(indexes, int.__lt__, value)
        return self

    def _add_condition(self, indexes, operator: Callable, value: int):
        indexes = (indexes,) if isinstance(indexes, int) else tuple(indexes)
        if operator not in self._conditions:
            self._conditions[operator] = []
        self._conditions[operator].append((indexes, value))

    def __call__(self, state: BasicState) -> bool:
        for operator, cond in self._conditions.items():
            for indexes, value in cond:
                s = 0
                for i in indexes:
                    s += state[i]
                if not operator(s, value):
                    return False
        return True

    def __repr__(self):
        strlist = []
        for operator, cond in self._conditions.items():
            operator_str = [o for o in self._OPERATOR if self._OPERATOR[o] == operator][0]
            for indexes, value in cond:
                strlist.append(f"{list(indexes)}{operator_str}{value}")
        return "&".join(strlist)

    def __eq__(self, other):
        return self._conditions == other._conditions

if __name__ == "__main__":
    def ps_fn(s) -> bool:
        return s[0] == 0 and (s[1] + s[2]) == 1 and (s[3] + s[4]) == 1 and s[5] == 0

    ps1 = PostSelect("[0]==0 & [1,2]==1 & [3,4]==1 & [5]==0")

    bs = BasicState([0,1,0,1,0,0])
    import time
    tfn_start = time.time()
    for _ in range(100000):
        ps_fn(bs)
    tfn_dur = time.time() - tfn_start

    tps_start = time.time()
    for _ in range(100000):
        ps1(bs)
    tps_dur = time.time() - tps_start

    print("Function:",tfn_dur)
    print("Object:",tps_dur)
