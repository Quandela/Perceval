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

import random
import sympy as sp


class Parameter:
    r"""A Parameter is a used as a variable in a circuit definition

    Parameters are a simple way to introduce named variables in a circuit. They take floating number values. When non
    defined they are associated to sympy symbols and will be used to perform symbolic calculations.

    :param name: name of the parameter
    :param value: optional value, when the value is provided at initialization, the parameter is considered as `fixed`
    :param min_v: minimal value that the parameter can take, is used in circuit optimization
    :param max_v: maximal value that the parameter can take, is used in circuit optimization
    :param is_expression: for symbolic parameter, the value is an expression to evaluate with context values
    """
    _id = 0

    def __init__(self, name: str, value: float = None,
                 min_v: float = None, max_v: float = None, periodic=True,
                 is_expression: bool = False):
        if min_v is not None:
            self._min = float(min_v)
        else:
            self._min = None
        if max_v is not None:
            self._max = float(max_v)
        else:
            self._max = None
        if value is None:
            self._symbol = sp.symbols(name, real=True)
            self._value = None
        else:
            if not isinstance(value, sp.Expr):
                self._value = self._check_value(value, self._min, self._max, periodic)
            else:
                self._value = value
            self._symbol = None
        self.name = name
        self._periodic = periodic
        self._pid = Parameter._id
        self._is_expression = is_expression
        self._original = None
        Parameter._id += 1

    @property
    def spv(self) -> sp.Expr:
        r"""The current value of the parameter defined as a sympy expression
        """
        if self._value is not None:
            return sp.S(self._value)
        else:
            return self._symbol

    @property
    def is_variable(self) -> bool:
        r""""Returns True for a non-fixed parameter"""
        return self._symbol is not None


    def __float__(self):
        r"""Convert the parameter to float, will fail if the parameter has no defined value
        """
        return float(self._value)

    def evalf(self, subs: dict = None):
        r"""Convert the parameter to float, will fail if the parameter has no defined value
        """
        if subs is None or not isinstance(self._value, sp.Expr):
            return float(self._value)
        return self._value.evalf(subs=subs)

    def is_symbolic(self):
        return self._value is None or isinstance(self._value, sp.Expr)

    def random(self):
        if self._symbol is None:
            return float(self._value)
        if self._min is not None and self._max is not None:
            return float(random.random() * (self._max-self._min) + self._min)
        return random.random()

    @staticmethod
    def _check_value(v: float, min_v: float, max_v: float, periodic: bool):
        if periodic and min_v is not None and max_v is not None:
            if v > max_v:
                p = int((v-max_v)/(max_v-min_v))
                v = v - (p+1) * (max_v-min_v)
            elif v < min_v:
                p = int((min_v-v)/(max_v-min_v))
                v = v + (p+1) * (max_v-min_v)
        if (min_v is not None and v < min_v) or (max_v is not None and v > max_v):
            raise ValueError("value %f out of bound [%f,%f]" %(v, min_v, max_v))
        return v

    def check_value(self, v):
        return self._check_value(v, self._min, self._max, self._periodic)

    def set_value(self, v: float, force: bool = False):
        r"""Define the value of a non-fixed parameter

        :param v: the value
        :param force: enable to set a fixed parameter
        :raise: `RuntimeError` if the parameter is fixed
        """
        v = self._check_value(v, self._min, self._max, self._periodic)
        if self.fixed and not force:
            raise RuntimeError("cannot set fixed parameter", v, self._value)
        self._value = v

    def fix_value(self, v):
        r"""Fix the value of a non-fixed parameter

        :param v: the value
        """
        self._symbol = None
        self._value = self._check_value(v, self._min, self._max, self._periodic)

    def reset(self):
        r"""Reset the value of a non-fixed parameter"""
        if self._symbol:
            self._value = None

    @property
    def defined(self) -> bool:
        r"""Return True if the parameter has a value (fixed or non fixed)
        """
        return self._value is not None

    @property
    def is_periodic(self) -> bool:
        r"""Return True if the parameter is defined as a period parameter
        """
        return self._periodic

    def set_periodic(self, periodic):
        r"""set periodic flag"""
        self._periodic = periodic

    @property
    def fixed(self) -> bool:
        r"""Return True if the parameter is fixed
        """
        return self._symbol is None and not self._is_expression

    def __repr__(self):
        return "Parameter(name='%s', value=%s%s%s)" % (str(self.name), str(self._value),
                                                       self._min is not None and ", min_v="+str(self._min) or "",
                                                       self._max is not None and ", max_v="+str(self._max) or "")

    @property
    def bounds(self) -> tuple[float, float]:
        r"""Minimal and maximal values for the parameter
        """
        return self._min, self._max

    @property
    def min(self):
        r"""The minimal value of the parameter
        """
        return self._min

    @min.setter
    def min(self, m: float):
        r"""Set the minimal value of the parameter
        """
        self._min = m

    @property
    def max(self) -> float:
        r"""The maximal value of the parameter

        """
        return self._max

    @max.setter
    def max(self, m: bool):
        r"""Set the maximal value of the parameter

        """
        self._max = m

    @property
    def pid(self):
        r"""Unique identifier for the parameter"""
        return self._pid

    def __mul__(self, other):
        if isinstance(other, Expression):
            return Expression(f"({self.name}*{other.name})", {self} | other.params)
        elif isinstance(other, Parameter):
            return Expression(f"({self.name}*{other.name})", {self, other})
        elif isinstance(other, (int, float)):
            return Expression(f"({other}*{self.name})", {self})
        raise TypeError("Unsupported parameter operation.")

    def __rmul__(self, other):
        return self.__mul__(other)

    def __add__(self, other):
        if isinstance(other, Expression):
            return Expression(f"({self.name}+{other.name})", {self} | other.params)
        elif isinstance(other, Parameter):
            return Expression(f"({self.name}+{other.name})", {self, other})
        elif isinstance(other, (int, float)):
            return Expression(f"({self.name}+{other})", {self})
        raise TypeError("Unsupported parameter operation.")

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, Expression):
            return Expression(f"({self.name}-{other.formula})", {self} | other.params)
        elif isinstance(other, Parameter):
            return Expression(f"({self.name}-{other.name})", {self, other})
        elif isinstance(other, (int, float)):
            return Expression(f"({self.name}-{other})", {self})
        raise TypeError("Unsupported parameter operation.")

    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            return Expression(f"({other}-{self.name})", {self})
        raise TypeError("Unsupported operation.")

    def __truediv__(self, other):
        if isinstance(other, Expression):
            return Expression(f"({self.name}/{other.name})", {self} | other._params)
        elif isinstance(other, Parameter):
            return Expression(f"({self.name}/{other.name})", {self, other})
        elif isinstance(other, (int, float)):
            return Expression(f"({self.name}/{other})", {self})
        raise TypeError("Unsupported parameter operation.")

    def __pow__(self, other):
        if isinstance(other, Expression):
            return Expression(f"({self.name}^{other.name})", {self} | other._params)
        elif isinstance(other, Parameter):
            return Expression(f"({self.name}^{other.name})", {self, other})
        elif isinstance(other, (int, float)):
            return Expression(f"({self.name}^{other})", {self})
        raise TypeError("Unsupported parameter operation.")

    def __neg__(self):
        # Ensure using __neg__ twice in a row returns the original parameter
        if self._original is not None:
            return self._original

        if isinstance(self, Expression):
            expr = Expression(f"(-{self.name})", self._params)
        else:
            expr = Expression(f"(-{self.name})", {self})
        expr._original = self
        return expr

class Expression(Parameter):
    """
    This class allows arithmetic manipulation of the Parameter class.
    A logical string is passed and the parameters with a corresponding name are created.
    Alternatively, one can specify the pre-defined parameters.

    :param name: string specifying equation, acts as name of Expression parameter.
    :param parameters: specifies the identities of existing parameters present in the expression name
    """
    def __init__(self, name: str, parameters: set[Parameter] = None):
        try:
            e = sp.S(name)
            self.name = f"({e})"
        except Exception as err:
            raise ValueError(f"{name} is not an expression: {err}")
        if not isinstance(e, sp.Expr):
            raise ValueError (f"{name} is not an expression")

        # Create set containing all parent parameters
        self._params = set() if parameters is None else set(parameters)
        super().__init__(self.name, is_expression=True, periodic=False)
        self._symbol = sp.S(name)

    def __repr__(self):
        return f"Expression({self.name[1:-1]}, parameters={self._params})"

    def __mul__(self, other):
        if isinstance(other, Expression):
            return Expression(f"({self.name}*{other.name})", other._params | self._params)
        elif isinstance(other, Parameter):
            return Expression(f"({self.name}*{other.name})", {other} | self._params)
        elif isinstance(other, (int, float)):
            return Expression(f"({other}*{self.name})", self._params)
        raise TypeError("Unsupported parameter operation.")

    def __rmul__(self, other):
        return self.__mul__(other)

    def __add__(self, other):
        if isinstance(other, Expression):
            return Expression(f"({self.name}+{other.name})", other._params | self._params)
        elif isinstance(other, (Parameter, P)):
            return Expression(f"({self.name}+{other.name})", {other} | self._params)
        elif isinstance(other, (int, float)):
            return Expression(f"({self.name}+{other})", self._params)
        raise TypeError("Unsupported parameter operation.")

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, Expression):
            return Expression(f"({self.name}-{other.name})", other._params | self._params)
        elif isinstance(other, Parameter):
            return Expression(f"({self.name}-{other.name})", {other} | self._params)
        elif isinstance(other, (int, float)):
            return Expression(f"({self.name}-{other})", self._params)
        raise TypeError("Unsupported parameter operation.")

    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            return Expression(f"({other}-{self.name})", self._params)
        raise TypeError("Unsupported operation.")

    def __truediv__(self, other):
        if isinstance(other, Expression):
            return Expression(f"({self.name}/{other.name})", other._params | self._params)
        elif isinstance(other, Parameter):
            return Expression(f"({self.name}/{other.name})", {other} | self._params)
        elif isinstance(other, (int, float)):
            return Expression(f"({self.name}/{other})", self._params)
        raise TypeError("Unsupported parameter operation.")

    def __pow__(self, other):
        if isinstance(other, (int, float)):
            return Expression(f"{self.name}^{other}", self._params)
        raise TypeError("Unsupported operation.")

    def __float__(self):
        """Updates Expression with respect to any changes made to parent Parameters"""
        if any(not param.defined for param in self._params):
            raise ValueError("Expression is symbolic, cannot compute its numerical value")
        return sp.S(self.name).subs({param.name : param._value for param in self._params})

    @property
    def parameters(self) -> list[Parameter]:
        """Returns list of parent parameters in alphabetical order"""
        return sorted(self._params, key=lambda obj: obj.name)

    @property
    def is_periodic(self) -> bool:
        """Expressions are not considered periodic"""
        return False


P = Parameter
E = Expression
