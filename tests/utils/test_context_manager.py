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

import pytest

from perceval import ContextManager, ContextManagerDecorator, encapsulate_managers, encapsulate_manager_list


def test_manager_basic():
    a = []

    def enter_manager():
        a.append(1)
        return a

    def exit_manager():
        a.append(2)

    cm = ContextManager(enter_manager, exit_manager)
    assert cm
    with cm as manager:
        assert a == [1]
        assert manager is a

    assert a == [1, 2]

    cm = ContextManager()
    assert not cm
    with cm as manager:
        assert manager is cm


def test_manager_exit():

    def exit_manager(exc_type, exc_val, exc_tb):
        if exc_type is ValueError:
            return True

    cm = ContextManager(at_exit=exit_manager)
    error = AssertionError("Should not be caught by the manager")

    with pytest.raises(AssertionError):
        with cm:
            raise AssertionError("Should not be caught by the manager")

    assert cm.exc_type is AssertionError
    assert str(cm.exc_val) == str(error)

    error = ValueError("Should be caught by the manager")
    with cm:
        raise error

    assert cm.exc_type is ValueError
    assert str(cm.exc_val) == str(error)


def test_manager_decorator_basic():
    a = []

    def enter_manager(v):
        a.append(v)
        return v - 1

    def exit_manager(v):
        a.append(v)

    cm_int = ContextManager(lambda: enter_manager(2), lambda: exit_manager(4))
    cm_ext = ContextManager(lambda: enter_manager(1), lambda: exit_manager(5))
    with encapsulate_managers(cm_ext, cm_int) as cm:
        assert cm == [0, 1]
        assert a == [1, 2]
        a.append(3)

    assert a == [1, 2, 3, 4, 5]

    a = []
    with ContextManagerDecorator(ContextManager(), lambda: enter_manager(1), lambda: exit_manager(5)) as cm:
        assert cm == [0]  # Empty manager ignored
        assert a == [1]

    assert a == [1, 5]

    a = []

    cm_int = encapsulate_managers(cm_ext, cm_int)
    cm_ext = ContextManager(lambda: enter_manager(0), lambda: exit_manager(6))
    with encapsulate_managers(cm_ext, cm_int) as cm:
        assert cm == [-1, 0, 1]  # Concatenated into a single list
        assert a == [0, 1, 2]
        a.append(3)

    assert a == [0, 1, 2, 3, 4, 5, 6]


def test_manager_decorator_error():
    def exit_manager(catch: bool, exc_type, exc_val, exc_tb):
        return catch

    cm_int = ContextManager(at_exit=lambda *args: exit_manager(False, *args))
    cm_ext = ContextManagerDecorator(cm_int, at_exit=lambda *args: exit_manager(False, *args))
    error = AssertionError("Should not be caught by the manager")
    with pytest.raises(AssertionError):
        with cm_ext:
            raise AssertionError("Should not be caught by the manager")

    assert cm_int.exc_type is type(error)
    assert str(cm_int.exc_val) == str(error)
    assert cm_ext.exc_type is type(error)
    assert str(cm_ext.exc_val) == str(error)

    cm_ext = ContextManagerDecorator(cm_int, at_exit=lambda *args: exit_manager(True, *args))
    error = ValueError("Should be caught by the manager")
    with cm_ext:
        raise error

    assert cm_ext.exc_type is type(error)
    assert str(cm_ext.exc_val) == str(error)

    cm_int = ContextManager(at_exit=lambda *args: exit_manager(True, *args))
    cm_ext = ContextManagerDecorator(cm_int, at_exit=lambda *args: exit_manager(False, *args))

    error = RuntimeError("Should also be caught by the manager")
    with cm_ext:
        raise error

    assert cm_int.exc_type is type(error)
    assert str(cm_int.exc_val) == str(error)
    assert cm_ext.exc_type is None


def test_manager_decorator_error_by_inner():

    def exit_manager(catch: bool, exc_type, exc_val, exc_tb):
        return catch

    def raise_error(error: Exception):
        raise error

    error = ValueError("Should be caught by the manager")
    cm_int = ContextManager(at_exit=lambda: raise_error(error))
    cm_ext = ContextManagerDecorator(cm_int, at_exit=lambda *args: exit_manager(True, *args))

    a  = 0
    with cm_ext:
        a = 1

    assert a == 1
    assert cm_ext.exc_type is type(error)
    assert str(cm_ext.exc_val) == str(error)

    error = RuntimeError("Should also be caught by the manager")
    cm_int = ContextManager(at_enter=lambda: raise_error(error))  # Same but at enter
    cm_ext = ContextManagerDecorator(cm_int)  # Can't catch the error here as we need not enter the body

    a  = 0
    with pytest.raises(RuntimeError):
        with cm_ext:
            a = 1

    assert a == 0
    assert cm_ext.exc_type is type(error)
    assert str(cm_ext.exc_val) == str(error)


def test_encapsulate_managers():
    a = []
    def make_manager(v):
        def enter_manager():
            a.append(v)
        return ContextManager(enter_manager)

    managers = [make_manager(i) for i in range(5)]
    full_manager = encapsulate_manager_list(managers)
    with full_manager:
        pass

    assert a == [0, 1, 2, 3, 4]

    empty_manager = encapsulate_manager_list([])
    assert not empty_manager
