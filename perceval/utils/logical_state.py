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

from typing import List


class LogicalState(list):
    def __init__(self, state: List[int] or str = None):
        """Represent a Logical state

        :param state: Can be either None, a list or a str, defaults to None
        :raises ValueError: Must have only 0 and 1 in a state
        :raises TypeError: Supports only None, list or str as state type
        """
        if state is None:
            super().__init__([])
            return
        if isinstance(state, str):
            state = [int(elem) for elem in state]
        if isinstance(state, list):
            if state.count(0) + state.count(1) != len(state):
                raise ValueError("A logical state should only contain 0s and 1s")
            super().__init__(state)
            return
        raise TypeError(f"LogicalState can be initialise with None, list or str, here {type(state)}")

    def __add__(self, other):
        temp = self.copy()
        temp.extend(other)
        return temp

    def __str__(self):
        if not self:
            return ""
        return ''.join([str(x) for x in self])


def generate_all_logical_states(n : int) -> List[LogicalState]:
    format_str = f"#0{n+2}b"
    logical_state_list = []
    for i in range(2**n):
        states = format(i, format_str)[2:]
        logical_state_list.append(LogicalState([int(state) for state in states]))
    return logical_state_list
