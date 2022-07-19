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

from typing import Dict, Callable

from perceval.components import Circuit


class PredefinedCircuit:
    def __init__(self,  c: Circuit,
                 name: str = None,
                 description: str = None,
                 heralds: Dict[int, int] = None,
                 post_select_fn: Callable = None):
        r"""Define a `PredefinedCircuit` which is a readonly circuit with more information about its usage

        :param c:
        :param name:
        :param description:
        :param heralds:
        """
        self._c = c
        self._name = name
        self._description = description
        self._heralds = heralds
        self._post_select_fn = post_select_fn

    @property
    def circuit(self):
        return self._c.copy()

    @property
    def description(self):
        return self._description

    @property
    def name(self):
        return self._name

    @property
    def heralds(self) -> dict:
        return self._heralds

    @property
    def has_post_select(self) -> bool:
        return self._post_select_fn is not None

    def post_select(self, s) -> bool:
        if self._post_select_fn is None:
            return True
        return self._post_select_fn(s)
