import sympy as sp
from .format import simple_float

from typing import Tuple

class Parameter:
    r"""A Parameter is a used as a variable in a circuit definition

    Parameters are a simple way to introduce named variables in a circuit. They take floating number values. When non
    defined they are associated to sympy symbols and will be used to perform symbolic calculations.

    :param name: name of the parameter
    :param value: optional value, when the value is provided at initialization, the parameter is considered as `fixed`
    :param min_v: minimal value that the parameter can take, is used in circuit optimization
    :param max_v: maximal value that the parameter can take, is used in circuit optimization
    """
    _id = 0

    def __init__(self, name: str, value: float = None, min_v: float = None, max_v: float = None):
        if value:
            value, _ = simple_float(value)
        self._value = value
        if value is None:
            self._symbol = sp.symbols(name, real=True)
        else:
            self._symbol = None
        self.name = name
        self._min = min_v
        self._max = max_v
        self._pid = Parameter._id
        Parameter._id += 1

    @property
    def spv(self) -> sp.Expr:
        r"""The current value of the parameter defined as a sympy expression
        """
        if self._value is not None:
            return sp.S(self._value)
        else:
            return self._symbol

    def __float__(self):
        r"""Convert the parameter to float, will fail if the parameter has no defined value
        """
        return float(self._value)

    def set_value(self, v: float):
        r"""Define the value of a non-fixed parameter

        :param v: the value
        :raise: `RuntimeError` if the parameter is fixed
        """
        if self.fixed:
            raise RuntimeError("cannot set fixed parameter")
        self._value = v

    def fix_value(self, v):
        r"""Fix the value of a non-fixed parameter

        :param v: the value
        """
        self._symbol = None
        self._value = v

    @property
    def defined(self) -> bool:
        r"""Return True if the parameter has a value (fixed or non fixed)
        """
        return self._value is not None

    @property
    def fixed(self) -> bool:
        r"""Return True if the parameter is fixed
        """
        return self._symbol is None

    def __repr__(self):
        return "Parameter(name='%s', value=%s%s%s)" % (str(self.name), str(self._value),
                                                       self._min is not None and ", min="+str(self._min) or "",
                                                       self._max is not None and ", max="+str(self._max) or "")

    @property
    def bounds(self) -> Tuple[float, float]:
        r"""Minimal and maximal values for the parameter
        """
        return (self._min, self._max)

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


P = Parameter
