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

import sys
from types import TracebackType
from typing import Callable, Iterable, Any

from .inspection import has_arguments


class ContextManager:
    """
    Class to be used with the keyword :code:`with` with custom, externally given :code:`__enter__` and :code:`__exit__` methods.

    The :code:`at_enter` method takes no parameter, and may return a value that can be retrieved using the :code:`as` keyword.
    Returning None will give the ContextManager when using :code:`as` instead.

    The :code:`at_exit` method can take no parameters, or take the arguments for a normal :code:`__exit__` method (beware they can be None).
    If it returns True, any exception that occurred is considered to be taken as if it was caught by a try except block.

    Example:

    >>> with ContextManager(lambda: print("Entered"), lambda: print("Exited")):
    >>>        print("In context manager")
    """

    def __init__(self,
                 at_enter: Callable[[], Any | None] = None,
                 at_exit: Callable[[], bool | None] | Callable[[type |None, Exception | None, TracebackType | None], bool | None] = None):
        self._at_enter = at_enter
        self._at_exit = at_exit
        self._has_arguments = has_arguments(self._at_exit) if self._at_exit is not None else False

        self.exc_type = None  # So we can access them afterward from the outside if needed
        self.exc_val = None
        self.exc_tb = None

    def __enter__(self):
        if self._at_enter is not None:
            res = self._at_enter()
            if res is not None:
                return res
        return self

    def __exit__(self, exc_type: type | None, exc_val: Exception | None, exc_tb: TracebackType | None) -> bool | None:
        self.exc_type = exc_type
        self.exc_val = exc_val
        self.exc_tb = exc_tb

        if self._at_exit is not None:
            if self._has_arguments:
                return self._at_exit(exc_type, exc_val, exc_tb)
            else:
                return self._at_exit()
        return None

    def __bool__(self):
        return self._at_enter is not None or self._at_exit is not None


class _DecoratorList(list):
    pass


class ContextManagerDecorator(ContextManager):
    """
    Context manager that encapsulate a :code:`ContextManager` with new enter and exit methods.
    The inner context manager will be resolved after this decorator start method and before its end method

    The value got when using the :code:`as` keyword is the list of values got from each individual context manager,
    where the external decorators are inserted first.

    The inner context manager can raise new exceptions in its exit, which can be caught by this layer, or will be raised again

    Example:
        Suppose A, B, C, and D are callables that print respectively A, B, C, and D

        >>> cm = ContextManager(A, B)
        >>> decorated_cm = ContextManagerDecorator(cm, C, D)
        >>> with decorated_cm:
        >>>    print("inside context manager")
        >>> # Printed: "C", "A", "inside context manager", "B", "D"

    """

    def __init__(self, sub_context: ContextManager, at_enter = None, at_exit = None):
        self._sub_context = sub_context
        super().__init__(at_enter, at_exit)

    def __enter__(self):
        sub_res = super().__enter__()
        if sub_res is None:
            sub_res = self

        res = _DecoratorList([sub_res])
        try:
            sub_res = self._sub_context.__enter__()
            if sub_res is self._sub_context:
                if self._sub_context:
                    res.append(self._sub_context)
            elif isinstance(sub_res, _DecoratorList):
                res += sub_res
            else:
                res.append(sub_res)

        except:
            super().__exit__(*sys.exc_info())  # We must still exit here since we entered
            # We have to exit here so we don't enter inside the managed body
            # Note: this prevents any upper layer to handle the exception - this is a python limit
            raise
        return res

    def __exit__(self, exc_type: type | None, exc_val: Exception | None, exc_tb: TracebackType | None):
        error_handled = False
        try:
            error_handled = self._sub_context.__exit__(exc_type, exc_val, exc_tb)
        except:
            # Replace by error raised by the subcontext __exit__ - May hide an already existing exception
            exc_type, exc_val, exc_tb = sys.exc_info()

        if error_handled:
            super().__exit__(None, None, None)
        else:
            error_handled = super().__exit__(exc_type, exc_val, exc_tb)

        if not error_handled and exc_val is not None:
            raise exc_val

        return error_handled

    def __bool__(self):
        return bool(self._sub_context) or super().__bool__()


def encapsulate_managers(outer_manager: ContextManager, inner_manager: ContextManager) -> ContextManager:
    """
    Combines two context managers, so that the inner manager is called in-between the outer manager enter and exit methods.
    :param outer_manager: The manager that will enter first and exit last
    :param inner_manager: The manager that will enter last and exit first
    :return: A new context manager that combines the two context managers
    """
    if not isinstance(inner_manager, ContextManager):
        enter_method = inner_manager.__enter__ if hasattr(inner_manager, "__enter__") else None
        exit_method = inner_manager.__exit__ if hasattr(inner_manager, "__exit__") else None
        inner_manager = ContextManager(enter_method, exit_method)
    return ContextManagerDecorator(inner_manager, outer_manager.__enter__, outer_manager.__exit__)


def encapsulate_manager_list(managers: Iterable[ContextManager]) -> ContextManager:
    """
    Combines many context managers, so that the first given manager is the outermost manager, and the last one the innermost one.
    :param managers: The managers that will be combined into one
    :return: A new context manager that combines all given managers (as a ContextManagerDecorator if the given list is not empty)
    """
    manager = ContextManager()  # Allow empty list
    for sub_manager in managers:
        manager = encapsulate_managers(manager, sub_manager)

    return manager
