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
from typing import Union
import sympy as sp

from perceval.utils.parameter import Parameter


class AComponent(ABC):
    DEFAULT_NAME = ''

    def __init__(self, m: int, name: str = None):
        self._m = m
        self._name = name

    @property
    def m(self) -> int:
        return self._m

    @property
    def name(self) -> str:
        if self._name is None:
            return self.DEFAULT_NAME
        return self._name


class AParametrizedComponent(AComponent, ABC):
    def __init__(self, m: int, name: str = None):
        super().__init__(m, name)
        self._params = {}
        self._vars = {}

    @property
    def vars(self) -> dict[str, Parameter]:
        return {p.name: p for p in self._params.values() if not p.fixed}

    def assign(self,
               assign: dict = None):
        if assign is None:
            return
        vs = self.vars
        if isinstance(assign, dict):
            for k, v in assign.items():
                vs[k].set_value(v)

    @property
    def defined(self) -> bool:
        """
        check if all parameters of the circuit are fully defined
        """
        for _, p in self._params.items():
            if not p.defined:
                return False
        return True

    @property
    def params(self):
        return self._params.keys()

    def get_parameters(self, all_params: bool = False) -> list[Parameter]:
        """Return the parameters of the circuit

        :param all_params: if False, only returns the variable parameters
        :return: the list of parameters
        """
        return [v for v in self._params.values() if all_params or not v.fixed]

    def reset_parameters(self) -> None:
        for v in self._params.values():
            v.reset()

    def _set_parameter(self,
                       name: str,
                       p: Union[Parameter, float],
                       min_v: float,
                       max_v: float,
                       periodic: bool = True) -> Parameter:
        """
            Define a new parameter for the circuit, it can be an existing parameter that we recycle updating
            min/max value or a parameter defined by a value that we create on the fly
        :param name:
        :param p:
        :param min_v:
        :param max_v:
        :param periodic:
        :return:
        """
        if isinstance(p, Parameter):
            if min_v is not None:
                if p.min is None or min_v > p.min:
                    p.min = float(min_v)
            if max_v is not None:
                if p.max is None or max_v < p.max:
                    p.max = float(max_v)
            if p.name in self._vars:
                if p.pid != self._vars[p.name].pid:
                    raise RuntimeError("two parameters with the same name in the circuit")
            if periodic is not None:
                p.set_periodic(periodic)
            self._vars[p.name] = p
        else:
            p = Parameter(value=p, name=name, min_v=min_v, max_v=max_v, periodic=periodic)
        self._params[name] = p
        return p

    def variable_def(self, out_parameters: dict, k, pname, default_value, map_param_kid=None):
        if map_param_kid is None:
            map_param_kid = {}
        if self._params[k].defined:
            if default_value is None or self._params[k]._value != default_value:
                v = self._params[k]._value
                if isinstance(v, sp.Expr):
                    v = str(v)
                    out_parameters[pname] = v
                elif default_value is None or abs(v-float(default_value)) > 1e-6:
                    out_parameters[pname] = v
        else:
            out_parameters[pname] = map_param_kid[self._params[k]._pid]

    def get_variables(self, _=None):
        return {}

    def map_parameters(self):
        map_param_kid = {}
        for k, p in self._params.items():
            if not p.defined:
                map_param_kid[p._pid] = p.name
        return map_param_kid
