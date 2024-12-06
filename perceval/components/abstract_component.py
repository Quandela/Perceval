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
from __future__ import annotations

from abc import ABC
from collections.abc import Iterable
import sympy as sp
import copy

from perceval.utils.parameter import Parameter


class AComponent(ABC):
    DEFAULT_NAME = None

    def __init__(self, m: int, name: str = None):
        self._m = m
        self._name = name

    @property
    def m(self) -> int:
        return self._m

    @property
    def name(self) -> str:
        """Returns component name"""
        if self._name is None:
            return self.DEFAULT_NAME
        return self._name

    @name.setter
    def name(self, new_name: str = None) -> None:
        """Sets new component name"""
        self._name = new_name

    def is_composite(self) -> bool:
        """
        Returns True if the component is itself composed of subcomponents
        """
        return False


class AParametrizedComponent(AComponent):
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
    def params(self) -> Iterable[str]:
        """Returns a list of all variable parameter names in the component"""
        return self._params.keys()

    def param(self, param_name: str) -> Parameter:
        """Returns a `Parameter` object from its name"""
        return self._params[param_name]

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
                       p: Parameter | float,
                       min_v: float,
                       max_v: float,
                       periodic: bool = True) -> Parameter:
        """
        Define a new parameter for the circuit, it can be an existing parameter that we recycle updating
        min/max value or a parameter defined by a value that we create on the fly

        :param name: parameter name
        :param p: parameter instance or numerical value
        :param min_v: minimum numerical value (can be None)
        :param max_v: maximum numerical value (can be None)
        :param periodic: True if the value is periodic (e.g. for an angle)
        :return: The corresponding `Parameter` object
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

    def _populate_parameters(self, out_parameters: dict, pname: str, default_value: float = None):
        """
        Populate an in/out dictionary with a {parameter name: best value for display} couple, if needed.
        A value equal to the optional default value will not be injected in the dictionary.

        :param out_parameters: out dictionary, where key/value pairs are added.
        :param pname: parameter name to consider, in the component definition.
            e.g. to retrieve "phi0" from PS(phi=P("phi0")), ask for pname="phi", as it's the parameter name for a PS.
        :param default_value: optional default numerical value. None means no default value.
        """
        p = self._params[pname]
        if p.defined:
            if default_value is None or p._value != default_value:
                v = p._value
                if isinstance(v, sp.Expr):
                    out_parameters[pname] = str(v)
                elif default_value is None or abs(v - float(default_value)) > 1e-6:
                    out_parameters[pname] = v
        else:
            out_parameters[pname] = self._params[pname].name

    def get_variables(self):
        return {}

    def copy(self, subs: dict | list = None):
        nc = copy.deepcopy(self)

        if subs is None:
            for k, p in nc._params.items():
                if p.defined:
                    v = float(p)
                else:
                    v = None
                nc._params[k] = Parameter(p.name, v, p.min, p.max, p.is_periodic)
                nc.__setattr__("_"+k, nc._params[k])
        else:
            if isinstance(subs, list):
                subs = {p.name: p.spv for p in subs}
            for k, p in nc._params.items():
                name = p.name
                min_v = p.min
                max_v = p.max
                is_periodic = p.is_periodic
                if p._value is None:
                    p = p._symbol.evalf(subs=subs)
                else:
                    p = p.evalf(subs=subs)
                if not isinstance(p, sp.Expr) or isinstance(p, sp.Number):
                    nc._params[k] = Parameter(name, float(p), min_v, max_v, is_periodic)
                else:
                    nc._params[k] = Parameter(name, None, min_v, max_v, is_periodic)
        for k, p in nc._params.items():
            nc.__setattr__("_" + k, nc._params[k])

        return nc
